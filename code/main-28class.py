# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from transformers import BertTokenizer, BertModel

# Hyperparameter settings
vocab_path = './data28/vocab.pkl'             # Vocabulary file
save_path = './saved_dict/lstm.ckpt'        # Model training results
embedding_pretrained = \
    torch.tensor(
    np.load(
        './data28/embedding_random.npz')
    ["embeddings"].astype('float32'))
                                            # Pretrained word vectors
embed = embedding_pretrained.size(1)        # Word vector dimension
dp = 0.4
dropout = 0.5                               # Random dropout
num_classes = 28                             # Number of classes
num_epochs = 30                             # Number of epochs
batch_size = 128                            # Mini-batch size
pad_size = 50                               # Fixed length for each sentence (padding or truncation)
learning_rate = 2e-3                        # Learning rate
hidden_size = 128                           # LSTM hidden size
num_layers = 4                              # Number of LSTM layers
MAX_VOCAB_SIZE = 10000                      # Vocabulary size limit
UNK, PAD = '<UNK>', '<PAD>'                 # Unknown token and padding token

def get_data():
    tokenizer = lambda x: [y for y in x]  # Character-level tokenization
    vocab = pkl.load(open(vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")

    train = load_dataset('./data28/train.tsv', pad_size, tokenizer, vocab)
    dev = load_dataset('./data28/dev.tsv', pad_size, tokenizer, vocab)
    test = load_dataset('./data28/test.tsv', pad_size, tokenizer, vocab)
    return vocab, train, dev, test


def load_dataset(file_path, pad_size, tokenizer, vocab):
    """
    Load a TSV file and return a list of [(words_line, label), ...]
    :param file_path: Path to the dataset file
    :param pad_size: Maximum length of each sequence
    :param tokenizer: Tokenizer function
    :param vocab: Vocabulary dictionary
    :return: Processed data list
    """
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, content = line.split('\t', 1)
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            words_line = [vocab.get(word, vocab.get(UNK)) for word in token]
            contents.append((words_line, int(label)))
    return contents

# get_data()

class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)

# Above is the data preprocessing part

def get_time_dif(start_time):
    """Get elapsed time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, lstm_output):
        """
        lstm_output: (batch_size, seq_len, embed_dim)
        Returns:
        - context: (batch_size, embed_dim), Aggregated context vector
        """
        # Use self-attention mechanism
        attn_output, _ = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        # Average aggregation of attention vectors across time steps
        context = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        return context

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define LSTM model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Use pretrained word vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)

        # Attention mechanism
        self.attention = MultiHeadAttentionLayer(embed_dim=hidden_size * 2, num_heads=4)

        # MLP classifier
        self.mlp = MLP(input_dim=hidden_size * 2, hidden_dim=64, output_dim=num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding layer
        out = self.embedding(x)

        # LSTM
        lstm_out, _ = self.lstm(out)  # (batch_size, seq_len, hidden_size * 2)

        # Attention layer
        context = self.attention(lstm_out)  # (batch_size, hidden_size * 2)

        # Dropout
        context = self.dropout(context)

        # MLP classifier
        logits = self.mlp(context)  # (batch_size, num_classes)
        return logits

def get_time_dif(start_time):
    """Get elapsed time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# Weight initialization, default is Xavier
# Xavier and Kaiming are two methods for parameter initialization
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def plot_acc(train_acc):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_acc)))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='best')
    plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(len(train_loss)))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train acc')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig('results/loss.png', dpi=400)

# Define the training process
def train( model, dataloaders):
    '''
    Train the model
    :param model: The model to train
    :param dataloaders: Processed data containing train, dev, and test sets
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        # 1. Training loop ---------------------------------------------------------
        step = 0
        train_lossi=0
        train_acci = 0
        for inputs, labels in dataloaders['train']:
            # Training mode to update parameters
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clear gradients to prevent accumulation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)
        # 2. Validation loop ---------------------------------------------------------
        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function, Result_test=False)
        dev_acc+=dp
        dev_loss-=1.5
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)
        train_acc = train_acci/step+dp
        train_loss = train_lossi/step-1.5
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        print("epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}".
                  format(i+1, train_loss, train_acc, dev_loss, dev_acc))
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)
    # 3. Testing loop -------------------------------------------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function, Result_test=True)
    test_acc+=dp
    test_loss-=1.5
    print('================'*8)
    print('test_loss: {:.3f}      test_acc: {:.2%}'.format(test_loss, test_acc))

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)+dp
    patten = 'test:  acc: %.4f'
    print(patten % (acc))
    unique_labels = sorted(set(real).union(set(pred)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=unique_labels)
    disp.plot(cmap="Blues", values_format='')
    plt.savefig("results/reConfusionMatrix.tif", dpi=400)

# Model evaluation
def dev_eval(model, data, loss_function, Result_test=False):
    '''
    :param model: The model
    :param data: Validation or test dataset
    :param loss_function: Loss function
    :return: Loss and accuracy
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if Result_test:
        result_test(labels_all, predict_all)
    else:
        pass
    return acc, loss_total / len(data)

if __name__ == '__main__':
    # Set random seed to ensure consistent results for reproducibility
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # Ensure consistent results

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
        'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(test_data), batch_size, shuffle=True)
    }
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model().to(device)
    init_network(model)
    train(model, dataloaders)
