import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import pandas as pd

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


def _get_padded_features(x):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(x_) for x_ in x])
  padx = []
  for i in range(len(x)):
    pads = np.nan*np.ones((d - len(x[i]),) + x[i].shape[1:])
    padx.append(np.concatenate([x[i], pads]))
  return np.array(padx)

def _get_padded_targets(t):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(t_) for t_ in t])
  padt = []
  for i in range(len(t)):
    pads = np.nan*np.ones(d - len(t[i]))
    padt.append(np.concatenate([t[i], pads]))
  return np.array(padt)[:, :, np.newaxis]


def _forward_impute(x):
    """
    x is a n_individuals by n_features by n_timesteps 2d numpy array
    returns equivalent of pandas' ffill
    """
    x_imputed = x.copy() # dont want inplace

    for individual in range(x_imputed.shape[0]):
        for col in x_imputed[individual,:,:].T:
            nan_inds = np.argwhere(np.isnan(col))
            if nan_inds.size != 0: # otherwise it's already filled
                first_nan_ind = nan_inds[0][0]
                if first_nan_ind == 0:
                    raise ValueError() # it's all nan
                else: 
                    col[first_nan_ind:] = col[first_nan_ind-1] # fill the rest

    return x_imputed

def prepocess_training_data_from_lists(x, t, e, vsize=0.2, val_data=None, random_state=563):
    """RNNs require different preprocessing for variable length sequences"""


    x = _get_padded_features(x)
    t = _get_padded_targets(t)
    e = _get_padded_targets(e)

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)


    x = _forward_impute(x)
    t = _forward_impute(t)
    e = _forward_impute(e)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).float()
    t_train = torch.from_numpy(t_train).float()
    e_train = torch.from_numpy(e_train).float()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])

      x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

      x_train = x_train[:-vsize]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]

    else:

      x_val, t_val, e_val = val_data

      x_val = _get_padded_features(x_val)
      t_val = _get_padded_features(t_val)
      e_val = _get_padded_features(e_val)

      x_val = torch.from_numpy(x_val).float()
      t_val = torch.from_numpy(t_val).float()
      e_val = torch.from_numpy(e_val).float()

    return (x_train, t_train, e_train,
            x_val, t_val, e_val)

class SurvivalClusteringMachine(nn.Module):
    def __init__(self, inputdim, hidden, 
                k=4, lstm_layers=1, layers=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam',
               risks=1):
        super(SurvivalClusteringMachine, self).__init__()

        self.k = k
        self.dist = dist
        self.temp = float(temp)
        self.discount = float(discount)
        self.optimizer = optimizer
        self.risks = risks
        self.hidden = hidden

        if layers is None:
            layers = []
        self.layers = layers


        if len(layers) == 0:
            lastdim = hidden
        else:
            lastdim = layers[-1]

        self.embedding = nn.LSTM(inputdim, hidden, lstm_layers,
                                batch_first=True)


        self.act = nn.SELU()
        self.shape = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(self.k))
                                         for r in range(self.risks)})
        self.scale = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(self.k))
                                         for r in range(self.risks)})

        self.gate = nn.ModuleDict({str(r+1): nn.Sequential(
            nn.Linear(hidden*2, self.k, bias=False)
            ) for r in range(self.risks)})

        self.scaleg = nn.ModuleDict({str(r+1): nn.Sequential(
            nn.Linear(hidden*2, self.k, bias=True)
            ) for r in range(self.risks)})

        self.shapeg = nn.ModuleDict({str(r+1): nn.Sequential(
            nn.Linear(hidden*2, self.k, bias=True)
            ) for r in range(self.risks)})


    def represent(self, x):
        """
        Dimension of representation is twice hidden size, since we're
        using forwards and backwards LSTM pass...
        """
        xrep, (h_n, c_n) = self.embedding(x)
        xrep = torch.cat([xrep[:,0,:], xrep[:,-1,:]],1)  # here
        xrep = nn.ReLU6()(xrep)
        return xrep


    def autoencoding_loss(self, x):
        return
    
    def predictive_loss(self, x):
        return

    def forward(self, x, risk):
        xrep = self.represent(x)
        dim = x.shape[0]
        return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
            self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
            self.gate[risk](xrep)/self.temp)
    
    def get_shape_scale(self, risk='1'):
        return (self.shape[risk], self.scale[risk])

def _conditional_weibull_loss(model, x, t, e, elbo=True, risk='1'):

  alpha = model.discount
  shape, scale, logits = model.forward(x, risk)

  k_ = shape
  b_ = scale

  lossf = []
  losss = []

  for g in range(model.k):

    k = k_[:, g]
    b = b_[:, g]

    s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
    f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
    f = f + s

    lossf.append(f)
    losss.append(s)

  losss = torch.stack(losss, dim=1)
  lossf = torch.stack(lossf, dim=1)

  if elbo:
    
    lossg = nn.Softmax(dim=1)(logits)
    print(lossg.shape, lossf.shape)
    losss = lossg*losss
    lossf = lossg*lossf
    losss = losss.sum(dim=1)
    lossf = lossf.sum(dim=1)

  else:

    lossg = nn.LogSoftmax(dim=1)(logits)
    losss = lossg + losss
    lossf = lossg + lossf
    losss = torch.logsumexp(losss, dim=1)
    lossf = torch.logsumexp(lossf, dim=1)

  uncens = np.where(e.cpu().data.numpy() == int(risk))[0]
  cens = np.where(e.cpu().data.numpy() != int(risk))[0]
  ll = lossf[uncens].sum() + alpha*losss[cens].sum()

  return -ll/float(len(uncens)+len(cens))


def train_scm(model,
              x_train, t_train, e_train,
              #x_valid, t_valid, e_valid,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100):
    nbatches = int(x_train.shape[0]/bs)+1


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dics = []
    costs = []
    i = 0
    for i in range(n_iter):
        for j in range(nbatches):

            xb = x_train[j*bs:(j+1)*bs]
            tb = t_train[j*bs:(j+1)*bs]
            eb = e_train[j*bs:(j+1)*bs]

            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = 0
            for r in range(model.risks):
                loss += _conditional_weibull_loss(model,
                                        xb,
                                        tb,
                                        eb,
                                        elbo=elbo,
                                        risk=str(r+1))
            print ("Train Loss:", float(loss))
            loss.backward()
            optimizer.step()



def pretrain_dsm(model, t_train, e_train, t_valid, e_valid,
                 n_iter=10000, lr=1e-2, thres=1e-4):

  premodel = DeepSurvivalMachinesTorch(1, 1,
                                       dist=model.dist,
                                       risks=model.risks,
                                       optimizer=model.optimizer)
  premodel.double()

  optimizer = get_optimizer(premodel, lr)

  oldcost = float('inf')
  patience = 0
  costs = []
  for _ in tqdm(range(n_iter)):

    optimizer.zero_grad()
    loss = 0
    for r in range(model.risks):
      loss += unconditional_loss(premodel, t_train, e_train, str(r+1))
    loss.backward()
    optimizer.step()

    valid_loss = 0
    for r in range(model.risks):
      valid_loss += unconditional_loss(premodel, t_valid, e_valid, str(r+1))
    valid_loss = valid_loss.detach().cpu().numpy()
    costs.append(valid_loss)
    #print(valid_loss)
    if np.abs(costs[-1] - oldcost) < thres:
      patience += 1
      if patience == 3:
        break
    oldcost = costs[-1]

  return premodel



class temp_layer_f(nn.Module):
    def __init__(self, hidden_size=8):
        super(temp_layer_f, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(25, self.hidden_size, num_layers=1, batch_first=True)
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
        self.dense3 = nn.Linear(8, 1)
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

            return (yb, yb_preds)

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