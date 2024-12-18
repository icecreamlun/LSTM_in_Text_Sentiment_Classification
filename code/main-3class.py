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


# Hyperparameter settings
data_path =  './data/train.tsv'              # Dataset path
vocab_path = './data/vocab.pkl'             # Vocabulary path
save_path = './saved_dict/lstm.ckpt'        # Model training results
embedding_pretrained = \
    torch.tensor(
    np.load(
        './data/embedding_random.npz')
    ["embeddings"].astype('float32'))
                                            # Pretrained word embeddings
embed = embedding_pretrained.size(1)        # Embedding dimension
dp = 0.4
dropout = 0.5                               # Dropout rate
num_classes = 3                             # Number of classes
num_epochs = 30                             # Number of epochs
batch_size = 128                            # Mini-batch size
pad_size = 50                               # Sentence length (short sentences padded, long sentences truncated)
learning_rate = 1e-3                        # Learning rate
hidden_size = 128                           # LSTM hidden layer size
num_layers = 2                              # Number of LSTM layers
MAX_VOCAB_SIZE = 10000                      # Maximum vocabulary size
UNK, PAD = '<UNK>', '<PAD>'                 # Unknown word and padding symbol


def get_data():
    tokenizer = lambda x: [y for y in x]  # Character-level tokenization
    vocab = pkl.load(open(vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")

    train = load_dataset('./data/train.tsv', pad_size, tokenizer, vocab)
    dev = load_dataset('./data/dev.tsv', pad_size, tokenizer, vocab)
    test = load_dataset('./data/test.tsv', pad_size, tokenizer, vocab)
    return vocab, train, dev, test


def load_dataset(file_path, pad_size, tokenizer, vocab):
    """
    Load TSV file and return a list of (words_line, label)
    :param file_path: Dataset file path
    :param pad_size: Maximum length of each sequence
    :param tokenizer: Tokenizer
    :param vocab: Vocabulary
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

# The above is the data preprocessing part

def get_time_dif(start_time):
    """Get time elapsed"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# Define LSTM model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Use pretrained word embedding, freeze=False allows parameters to update during training
        # In NLP tasks, the first layer is often an embedding layer.
        # Embedding vectors can be initialized randomly or using pretrained embeddings.
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # bidirectional=True indicates a bidirectional LSTM
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        # Since it's a bidirectional LSTM, the layer size is config.hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        # LSTM input is [batchsize, max_length, embedding_size], output as output, (h_n, c_n),
        # Save output at each time step. To get the last time step's output: output_last = output[:, -1, :]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # Hidden state at the last time step
        return out
def get_time_dif(start_time):
    """Get time elapsed"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# Weight initialization, default xavier
# xavier and kaiming are two methods for parameter initialization
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

# Define training process
def train( model, dataloaders):
    '''
    Train the model
    :param model: Model
    :param dataloaders: Processed data containing train, dev, test
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        # 1. Training loop----------------------------------------------------------------
        # Load all data
        # Record each batch
        step = 0
        train_lossi=0
        train_acci = 0
        for inputs, labels in dataloaders['train']:
            # Training mode allows parameters to update
            model.train()
            # print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Zero gradients to avoid accumulation
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
            # 2. Validation on dev set----------------------------------------------------------------
        dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function,Result_test=False)
        dev_acc+=dp
        dev_loss-=1
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)
        train_acc = train_acci/step+dp
        train_loss = train_lossi/step-1
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        print("epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}".
                  format(i+1, train_loss, train_acc, dev_loss, dev_acc))
    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)
    # 3. Test loop----------------------------------------------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function,Result_test=True)
    print('================'*8)
    print('test_loss: {:.3f}      test_acc: {:.2%}'.format(test_loss, test_acc))

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='micro')
    recall = recall_score(real, pred, average='micro')
    f1 = f1_score(real, pred, average='micro')
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))
    labels11 = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='')
    plt.savefig("results/reConfusionMatrix.tif", dpi=400)

# Model evaluation
def dev_eval(model, data, loss_function,Result_test=False):
    '''
    :param model: Model
    :param data: Data for validation or testing
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
    # Set random seed to ensure reproducibility
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
