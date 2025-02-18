import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import gensim.downloader as api
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb

def preprocess_text(text):
    sentences = re.split(r'\.', text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    print(f"Total number of sentences: {len(sentences)}")
    processed_sentences = [simple_preprocess(sentence) for sentence in sentences]
    return sentences, processed_sentences

# Import file
main_corpus = "Auguste_Maquet.txt"
with open(main_corpus, 'r', encoding='utf-8') as f:
    text = f.read()
original_sentences, processed_corpus = preprocess_text(text)

# Pre-trained Word2Vec model from gensim
wv = api.load('word2vec-google-news-300')

# Vocabulary from all words in the corpus
all_words = [word for sentence in processed_corpus for word in sentence]
vocabulary = list(set(all_words))
vocabulary.append('<UNK>')
dictionary = Dictionary([vocabulary])

# Corpus to 5-gram embeddings
def get_word_embedding(word, model, embedding_dim=300):
    if word in model.key_to_index:
        return model[word]
    else:
        return np.zeros(embedding_dim)

def convert_to_5gram_embeddings(tokenized_sentences, model, embedding_dim=300, window_size=5):
    five_gram_embeddings = []
    next_words = []
    original_indices = []
    for idx, sentence in enumerate(tokenized_sentences):
        for i in range(len(sentence) - window_size):
            five_words = sentence[i:i + window_size]
            next_word = sentence[i + window_size]

            word_embeddings = [get_word_embedding(word, model, embedding_dim) for word in five_words]
            concatenated_embeddings = np.concatenate(word_embeddings)
            
            five_gram_embeddings.append(concatenated_embeddings)
            next_words.append(next_word)
            original_indices.append(idx)
    return five_gram_embeddings, next_words, original_indices

embedding_dim = 300
window_size = 5
# Tokenized corpus to embeddings
five_gram_embeddings, next_words, original_indices = convert_to_5gram_embeddings(processed_corpus, wv, embedding_dim, window_size)

# Splits
indices = np.arange(len(five_gram_embeddings))
train_val_indices, test_indices = train_test_split(indices, test_size=20000, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=10000, random_state=42)
train_embeddings = [five_gram_embeddings[i] for i in train_indices]
train_targets = [next_words[i] for i in train_indices]
train_orig_indices = [original_indices[i] for i in train_indices]
val_embeddings = [five_gram_embeddings[i] for i in val_indices]
val_targets = [next_words[i] for i in val_indices]
val_orig_indices = [original_indices[i] for i in val_indices]
test_embeddings = [five_gram_embeddings[i] for i in test_indices]
test_targets = [next_words[i] for i in test_indices]
test_orig_indices = [original_indices[i] for i in test_indices]

# Targets to indices
def targets_to_indices(targets, dictionary):
    return [dictionary.token2id.get(word, dictionary.token2id['<UNK>']) for word in targets]

train_targets_indices = targets_to_indices(train_targets, dictionary)
val_targets_indices = targets_to_indices(val_targets, dictionary)
test_targets_indices = targets_to_indices(test_targets, dictionary)

# DataLoader
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return embedding, target

def print_example_sentences(indices, num_examples=5, set_name="Set"):
    print(f"\nExample sentences from the {set_name}:")
    for i in range(num_examples):
        idx = indices[i]
        print(f"Example {i + 1}: {original_sentences[idx]}")

# Hyperparameter combinations
hyperparameter_combinations = [
    {'learning_rate': 0.0001, 'batch_size': 32, 'optimizer': 'Adam'},
    {'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'AdamW'},
    {'learning_rate': 0.0005, 'batch_size': 16, 'optimizer': 'Adam'}
]

def run_experiment(learning_rate, batch_size, optimizer_name):
    # Initialize W&B with a unique run name
    wandb.init(
        project='LM1', 

        name=f"lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_name}"
    )
    
    class LanguageModel(nn.Module):
        def __init__(self, input_size, hidden_size, vocab_size, dropout_prob=0.1):
            super(LanguageModel, self).__init__()
            self.hidden1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(p=dropout_prob)
            self.hidden2 = nn.Linear(hidden_size, vocab_size)
    
        def forward(self, x):
            x = torch.relu(self.hidden1(x))
            x = self.dropout(x)  
            x = self.hidden2(x)
            return x
    
    # Hyperparameters
    input_size = embedding_dim * window_size
    hidden_size = 300
    vocab_size = len(dictionary)
    num_epochs = 10
    
    # Log hyperparameters
    wandb.config.update({
        'input_size': input_size,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'optimizer': optimizer_name
    })
    
    # DataLoaders
    train_loader = DataLoader(EmbeddingDataset(train_embeddings, train_targets_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(val_embeddings, val_targets_indices), batch_size=1)
    test_loader = DataLoader(EmbeddingDataset(test_embeddings, test_targets_indices), batch_size=1)
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageModel(input_size, hidden_size, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Early stopping
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = f'/ssd_scratch/cvit/shaon/lm1/best_model_{optimizer_name}.pth'
    
    # Training
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Log metrics to W&B
        wandb.log({
            'Train Loss': avg_train_loss,
            'Validation Loss': avg_val_loss
        })
        
        # Check for early stopping and save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            wandb.save(best_model_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Plotting the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve for {optimizer_name} - LR: {learning_rate}, BS: {batch_size}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'learning_curve_{optimizer_name}.png')
    wandb.log({'Learning Curve': wandb.Image(f'learning_curve_{optimizer_name}.png')})
    plt.close()
    
    # Evaluate on test set
    def perplexity_mat(model, data_loader):
        model.eval()
        total_loss = 0
        total_words = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_words += targets.size(0)
        
        avg_loss = total_loss / total_words
        perplexity = np.exp(avg_loss)
        return perplexity

    test_perplexity = perplexity_mat(model, test_loader)
    print(f"Test Perplexity: {test_perplexity:.4f}")
    
    # Log perplexity metrics to W&B
    wandb.log({
        'Test Perplexity': test_perplexity
    })
    
    # Print example sentences
    print_example_sentences(train_orig_indices, set_name="Training Set")
    print_example_sentences(val_orig_indices, set_name="Validation Set")
    print_example_sentences(test_orig_indices, set_name="Test Set")
    
    # Close W&B run
    wandb.finish()

# Run experiments for each combination
for params in hyperparameter_combinations:
    run_experiment(params['learning_rate'], params['batch_size'], params['optimizer'])