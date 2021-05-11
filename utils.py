import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pandas as pd

def foo(x):
    return 43 + x


def prepocess_training_data_chirag(x, y, vsize, random_state):
    """RNNs require different preprocessing for variable length sequences"""

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)

    x_train, y_train = x[idx], y[idx]

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()


    vsize = int(vsize*x_train.shape[0])

    x_val, y_val = x_train[-vsize:], y_train[-vsize:]

    x_train = x_train[:-vsize]
    y_train = y_train[:-vsize]

    return (x_train, y_train,
            x_val, y_val)


class temp_layer_f(nn.Module):
    def __init__(self, hidden_size=8):
        super(temp_layer_f, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(35, self.hidden_size, num_layers=1, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
    def forward(self, x):
        lstm_output, (h_n, c_n) = self.lstm(x)
        x = lstm_output
        #x = self.dense(x)
        return x

class temp_layer_g(nn.Module):
    def __init__(self):    
        super(temp_layer_g, self).__init__()
        self.lstm = nn.LSTM(8, 8, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(8, 8)
        self.dense2 = nn.Linear(8, 8)
        self.dense3 = nn.Linear(8, 3)
        self.relu = nn.ReLU6()
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.lstm(x)[0] # can't we omit this from g?
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x

class temp_layer_h(nn.Module):
    def __init__(self, K=3):    
        super(temp_layer_h, self).__init__()
        self.dense1 = nn.Linear(8, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, K)
        self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.softmax(x)
        return x

class temp_layer_h_gumbo(nn.Module):
    def __init__(self, K=64):    
        super(temp_layer_h_gumbo, self).__init__()
        self.k = K
        self.dense_in = nn.Linear(8, K)
        #gumbo here
        self.dense_out = nn.Linear(K, 8)
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        x = self.dense_in(x)
        x = F.gumbel_softmax(x, tau=0.5, hard=True)
        x = self.dense_out(x)
        x = self.softmax(x)
        return x


def run_pretrain_loop(x_train, y_train, x_val, y_val, tf, tg, bs=128, n_epochs=50):
    nbatches = int(x_train.shape[0]/bs)+1

    optimizer = torch.optim.Adam(
            [{'params': tf.parameters()},
            {'params': tg.parameters()}]
            , lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    val_auc = 0.0

    pbar = tqdm.tqdm(range(n_epochs), position=0, leave=True)
    for i in pbar:
        for j in range(nbatches):
            xb = x_train[j*bs:(j+1)*bs]
            yb = y_train[j*bs:(j+1)*bs]
            
            yb_preds = tg(tf(xb))

            optimizer.zero_grad()
            loss = loss_fn(yb_preds, yb)
            loss.backward()
            optimizer.step()
            
            auc = roc_auc_score(yb.view(-1).detach().numpy(), yb_preds.view(-1).detach().numpy())
            loss_value = loss.detach().numpy()
            pbar.set_postfix_str("loss = {:1.3f}, auc = {:.2f}, val_auc = {:.2f}".format(
                loss_value,
                auc,
                val_auc))
        
        val_preds = tg(tf(x_val))
        val_auc = roc_auc_score(y_val.view(-1).detach().numpy(), val_preds.view(-1).detach().numpy())


def training_loop_iter_K(K, x_train, y_train, x_val, y_val, bs=128, n_epochs=250, n_epochs_pretrain=500, tf=None, tg=None):

    if tf == None or tg == None:
        tf = temp_layer_f()
        tg = temp_layer_g()

        run_pretrain_loop(x_train, y_train, x_val, y_val, tf, tg, bs=bs, n_epochs=n_epochs_pretrain)

    nbatches = int(x_train.shape[0]/bs)+1

    th_iter = temp_layer_h_gumbo(K=K)
    optimizer = torch.optim.Adam(
        [
            #{'params': tf.parameters()},
            #{'params': tg.parameters()},
            {'params': th_iter.parameters()}
        ]
        , lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    loss_fn_autoencoder = torch.nn.MSELoss()

    val_auc = 0.0
    val_auc_squeezed = 0.0

    metrics_record = pd.DataFrame(columns=["loss","auc","val_auc","auc_squeezed","val_auc_squeezed","loss_value_squeezed","val_loss_value_squeezed"],
                index=np.arange(0, n_epochs), dtype=np.float64)

    pbar = tqdm.tqdm(range(n_epochs), position=0, leave=True)
    for i in pbar:
        for j in range(nbatches):
            xb = x_train[j*bs:(j+1)*bs]
            yb = y_train[j*bs:(j+1)*bs]
            
            yb_preds = tg(tf(xb))
            yb_preds_squeezed = tg(th_iter(tf(xb)))

            optimizer.zero_grad()
            #loss = loss_fn(yb_preds, yb) + loss_fn(yb_preds_squeezed, yb)
            loss = loss_fn(yb_preds_squeezed, yb) + loss_fn_autoencoder(th_iter(tf(xb)), tf(xb))
            loss.backward()
            optimizer.step()
            
            auc = roc_auc_score(yb.view(-1).detach().numpy(),
                                yb_preds.view(-1).detach().numpy())
            
            auc_squeezed = roc_auc_score(yb.view(-1).detach().numpy(),
                                yb_preds_squeezed.view(-1).detach().numpy())    

            loss_value = loss.detach().double().numpy()
            loss_value_squeezed = loss_fn(yb_preds_squeezed, yb).detach().double().numpy()

            pbar.set_postfix_str("loss = {:1.3f}, auc = {:.2f}, val_auc = {:.2f}, auc_squeezed = {:.2f}, val_auc_squeezed = {:.2f}".format(
                loss_value,
                auc,
                val_auc,
                auc_squeezed,
                val_auc_squeezed))
        
        val_preds = tg(tf(x_val))
        val_preds_squeezed = tg(th_iter(tf(x_val)))

        val_auc = roc_auc_score(y_val.view(-1).detach().numpy(), val_preds.view(-1).detach().numpy())
        val_auc_squeezed = roc_auc_score(y_val.view(-1).detach().numpy(), val_preds_squeezed.view(-1).detach().numpy())

        val_loss_value_squeezed = loss_fn(val_preds_squeezed, y_val).detach().double().numpy()

        metrics_record.iloc[i] = [loss_value,
                auc,
                val_auc,
                auc_squeezed,
                val_auc_squeezed, 
                loss_value_squeezed,
                val_loss_value_squeezed]
    
    metrics_record['epoch'] = metrics_record.index + 1

    return metrics_record