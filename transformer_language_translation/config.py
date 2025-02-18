from pathlib import Path
import os

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 200,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 768,
        "num_layers" :8,
        "num_heads" : 12,
        "datasource": 'english_french.csv',
        "val_datasource": 'english_french_val.csv',
        "test_datasource": "english_french_test.csv",
        "lang_src": "English",
        "lang_tgt": "French",
        "model_folder": "weights",
        "model_basename": "t3model",
        "end" : "trans3",
        "preload": "latest",
        # "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "model_filename": "/ssd_scratch/cvit/shaon/trans2/t2model37.pt"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"/ssd_scratch/cvit/shaon/{config['end']}"
    os.makedirs(model_folder,exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"/ssd_scratch/cvit/shaon/{config['end']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])



### gnode023---> trans3, gnode057---> trans1, gnode033 --> t
