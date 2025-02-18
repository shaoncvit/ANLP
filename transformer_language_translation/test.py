from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

#import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import pandas as pd
import warnings
from tqdm import tqdm
import os
import wandb
from pathlib import Path

# Huggingface datasets and tokenizers
#from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from torchmetrics.text import BLEUScore
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
import random
import nltk
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer, cer



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def Bleu4(gt, pred):
    bleu4_score = sentence_bleu([nltk.word_tokenize(gt)], nltk.word_tokenize(pred), weights=(0.25, 0.25, 0.25, 0.25))
    return bleu4_score



def get_all_sentences(ds, lang):
    for item in ds[lang]:
        yield item

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    train_ds_raw = pd.read_csv(config['datasource'])
    test_ds_raw = pd.read_csv(config['test_datasource'])

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])
    #train_ds_raw = 
    # Keep 90% for training, 10% for validation
    # train_ds_size = int(0.9 * len(ds_raw))
    # val_ds_size = len(ds_raw) - train_ds_size
    # train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in train_ds_raw[config['lang_src']]:
        src_ids = tokenizer_src.encode(item).ids
        tgt_ids = tokenizer_tgt.encode(item).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt



def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'], num_layers = config['num_layers'] , num_heads = config['num_heads'])
    return model



def run_test(config, model, test_ds, tokenizer_src, tokenizer_tgt, max_len, device):
    
    
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []
    references = []

    #output file for validation result
    # output_sourcefile = os.path.join(model_basename,"validation_output.txt")
    # output_main_dirfile = "/home2/shaon/ANLP/transformer_from_scratch"

    # output_file = os.path.join(output_main_dirfile, output_sourcefile)

    output_file = "test_output3.txt"

    model_filename = config["model_filename"]

    with open(output_file, "w") as f:

        try:
            # get the console window width
            with os.popen('stty size', 'r') as console:
                _, console_width = console.read().split()
                console_width = int(console_width)
        except:
            # If we can't get the console width, use 80 as default
            console_width = 80

        with torch.no_grad():
            for batch in test_ds:
                count += 1
                encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(
                    0) == 1, "Batch size must be 1 for validation"

                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
                references.append([target_text])

                #calculate metrics using external libraries
                bleu_score = Bleu4(target_text, model_out_text)

                cer_score = cer(target_text, model_out_text)
                wer_score = wer(target_text, model_out_text)

                # Print and save to the file
                f.write('-' * console_width + '\n')
                f.write(f"SOURCE: {source_text}\n")
                f.write(f"TARGET: {target_text}\n")
                f.write(f"PREDICTED: {model_out_text}\n")
                f.write(f"BLEU Score: {bleu_score:.4f}, CER: {cer_score:.4f}, WER: {wer_score:.4f}\n")
                f.write('-' * console_width + '\n\n')






                # Print on console if needed
                print('-' * console_width)
                print(f"Taking inference on: {model_filename}")
                print(f"{f'SOURCE: ':>12}{source_text}")
                print(f"{f'TARGET: ':>12}{target_text}")
                print(f"{f'PREDICTED: ':>12}{model_out_text}")
                print(f"BLEU Score: {bleu_score:.4f}, CER: {cer_score:.4f}, WER: {wer_score:.4f}")




def test_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    device = torch.device(device)



    _, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = "/ssd_scratch/cvit/shaon/trans3/t3model34.pt"
    print(model_filename)
    # if model_filename:
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    print(state["model_state_dict"].keys())
    model.load_state_dict(state['model_state_dict'])
    
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
        
    # else:
    #     print('Model path is not available')
        
        
    run_test(config, model, test_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    test_model(config)

