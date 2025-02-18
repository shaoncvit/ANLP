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

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        embedded = self.dropout(self.embedding(x))
        output, _ = self.lstm(embedded)
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
        # val_perplexity = calculate_perplexity(model, val_loader, criterion, device)
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
        # print(f"Val Perplexity: {val_perplexity:.2f}")
        print(f"Test Perplexity: {test_perplexity:.2f}")
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"{epoch}.pt"))

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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    with open(args.corpus_file, 'r') as f:
        text = f.read()
    processed_corpus = preprocess_text(text)
    
    # Create dictionary and tokenize corpus
    dictionary = Dictionary(processed_corpus)
    args.vocab_size = len(dictionary)  # Set vocab_size based on the dictionary
    tokenized_corpus = [dictionary.doc2idx(doc) for doc in processed_corpus]
    
    # Split data
    train_val, test = train_test_split(tokenized_corpus, test_size=20000, random_state=42)
    train, val = train_test_split(train_val, test_size=10000, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = LanguageModelDataset(train)
    val_dataset = LanguageModelDataset(val)
    test_dataset = LanguageModelDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Create model, criterion, and optimizer
    model = LanguageModel(vocab_size=len(dictionary), 
                          embed_size=args.embed_size, 
                          hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, 
                          dropout=args.dropout)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Train and evaluate
    val_perplexity, test_perplexity = train_and_evaluate(args, model, train_loader, val_loader, test_loader, criterion, optimizer, device)
    
    print(f"Best validation perplexity: {val_perplexity:.2f}")
    print(f"Test perplexity: {test_perplexity:.2f}")
    
    # Load best model and write perplexity scores
    best_model = LanguageModel(vocab_size=len(dictionary), 
                               embed_size=args.embed_size, 
                               hidden_size=args.hidden_size, 
                               num_layers=args.num_layers, 
                               dropout=args.dropout)
    best_model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt")))
    best_model.to(device)
    
    train_perplexity_file_full_name = f"{args.roll_number}-LM2-train-perplexity.txt"
    val_perplexity_file_full_name = f"{args.roll_number}-LM2-val-perplexity.txt"
    test_perplexity_file_full_name = f"{args.roll_number}-LM2-test-perplexity.txt"

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    
    write_perplexity_scores(best_model, train_loader, dictionary, train_perplexity_file_full_name, device)
    write_perplexity_scores(best_model, val_loader, dictionary,val_perplexity_file_full_name, device )
    write_perplexity_scores(best_model, test_loader, dictionary, test_perplexity_file_full_name, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Training")
    parser.add_argument("--corpus_file", type=str, default="Auguste_Maquet.txt", help="Path to the corpus file")
    parser.add_argument("--embed_size", type=int, default=300, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=300, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--project_name", type=str, default="LSTM_model", help="Wandb project name")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to save the model")
    parser.add_argument("--roll_number", type=str, default="2023701018", help="File to save train and test perplexity scores")
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    main(args)