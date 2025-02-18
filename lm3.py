import torch
import re
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import wandb
import math
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import argparse
import os

def preprocess_text(text):
    sentences = re.split(r'\.', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    print("Total Number of sentences:", len(sentences))
    processed_sentences = [simple_preprocess(sentence) for sentence in sentences]
    return processed_sentences

class LanguageModelDataset(Dataset):
    def __init__(self, data):
        self.data = [seq for seq in data if len(seq) > 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    mask = (inputs_padded != 0).float()
    return inputs_padded, targets_padded, mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoderLite(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout):
        super(TransformerDecoderLite, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        tgt_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        output = self.transformer_decoder(src, src, tgt_mask=tgt_mask, memory_mask=None)
        output = output.transpose(0, 1)  # (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        output = self.dropout(output)
        output = self.fc(output)
        return output

def calculate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            outputs = model(inputs, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = (loss * mask.view(-1)).sum()
            total_loss += loss.item()
            total_words += mask.sum().item()
    return total_loss / total_words if total_words > 0 else float("inf")

def calculate_perplexity(model, loader, criterion, device):
    avg_loss = calculate_loss(model, loader, criterion, device)
    return math.exp(avg_loss)

def train_and_evaluate(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
    if args.use_wandb:
        wandb.init(project=args.project_name, config=vars(args))
    
    model.to(device)
    best_val_perplexity = float('inf')
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        for inputs, targets, mask in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs.view(-1, args.vocab_size), targets.view(-1))
            loss = (loss * mask.view(-1)).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_perplexity = calculate_perplexity(model, train_loader, criterion, device)

        val_loss = calculate_loss(model, val_loader, criterion, device)
        val_perplexity = math.exp(val_loss)
        test_perplexity = calculate_perplexity(model, test_loader, criterion, device)
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_perplexity": train_perplexity,
                "val_perplexity": val_perplexity,
                "test_perplexity": test_perplexity
            })
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
        print(f"Test Perplexity: {test_perplexity:.2f}")
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"epoch_{epoch}.pt"))

        if val_perplexity < best_val_perplexity:
            best_val_perplexity = val_perplexity
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pt"))
    
    if args.use_wandb:
        wandb.finish()
    
    return best_val_perplexity, test_perplexity

def write_perplexity_scores(model, loader, dictionary, filename, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with open(filename, 'w') as f:
        total_perplexity = 0
        count = 0
        with torch.no_grad():
            for inputs, targets, mask in loader:
                inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
                outputs = model(inputs, mask)
                for i in range(inputs.size(0)):
                    sentence = ' '.join([dictionary[idx.item()] for idx in inputs[i] if idx.item() != 0])
                    loss = criterion(outputs[i], targets[i])
                    if mask[i].sum().item() == 0:
                        continue
                    perplexity = math.exp((loss * mask[i]).sum().item() / mask[i].sum().item())
                    f.write(f"{sentence}\t{perplexity:.2f}\n")
                    total_perplexity += perplexity
                    count += 1
        average_perplexity = total_perplexity / count
        f.write(f"Average Perplexity: {average_perplexity:.2f}")
        print("Perplexity files are created")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.corpus_file, 'r') as f:
        text = f.read()
    processed_corpus = preprocess_text(text)
    
    dictionary = Dictionary(processed_corpus)
    args.vocab_size = len(dictionary)
    tokenized_corpus = [dictionary.doc2idx(doc) for doc in processed_corpus]
    
    train_val, test = train_test_split(tokenized_corpus, test_size=20000, random_state=42)
    train, val = train_test_split(train_val, test_size=10000, random_state=42)
    
    train_dataset = LanguageModelDataset(train)
    val_dataset = LanguageModelDataset(val)
    test_dataset = LanguageModelDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers = 1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    model = TransformerDecoderLite(vocab_size=len(dictionary), 
                                   d_model=args.d_model, 
                                   nhead=args.nhead, 
                                   num_layers=args.num_layers, 
                                   dropout=args.dropout)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    val_perplexity, test_perplexity = train_and_evaluate(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device)
    
    print(f"Best validation perplexity: {val_perplexity:.2f}")
    print(f"Test perplexity: {test_perplexity:.2f}")
    
    best_model = TransformerDecoderLite(vocab_size=len(dictionary), 
                                        d_model=args.d_model, 
                                        nhead=args.nhead, 
                                        num_layers=args.num_layers, 
                                        dropout=args.dropout)
    best_model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt")))


    best_model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    train_perplexity_file_full_name = f"{args.roll_number}-LM3-train-perplexity.txt"
    val_perplexity_file_full_name = f"{args.roll_number}-LM3-val-perplexity.txt"
    test_perplexity_file_full_name = f"{args.roll_number}-LM3-test-perplexity.txt"
    
    write_perplexity_scores(best_model, train_loader, dictionary, train_perplexity_file_full_name, device)
    write_perplexity_scores(best_model, val_loader, dictionary,val_perplexity_file_full_name, device )
    write_perplexity_scores(best_model, test_loader, dictionary, test_perplexity_file_full_name, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Decoder Lite Language Model Training")
    parser.add_argument("--corpus_file", type=str, default="Auguste_Maquet.txt", help="Path to the corpus file")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--project_name", type=str, default="Transformer_LM_Lite", help="Wandb project name")
    parser.add_argument("--model_dir", type=str, default="transformer_lite_model", help="Directory to save the model")
    parser.add_argument("--roll_number", type=str, default="2023701018", help="File to save train and test perplexity scores")
   
    
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    main(args)