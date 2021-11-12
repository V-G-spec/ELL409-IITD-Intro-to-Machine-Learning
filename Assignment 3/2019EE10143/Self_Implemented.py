# %%
"""
## Imports
"""

# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
random_state = 7
np.random.seed(random_state)

# %%
"""
## Load the Data
"""

# %%
#<Something>
file_name = '2019EE10143.csv'
split_frac = 0.8
tot_ex = 1500
    
def load_data(file_name, split_frac, tot_ex, random_state=random_state):
    df = pd.read_csv(file_name, header=None)
    cols = len(df.columns) #785
    num_ft = cols-1
    crop_frac = tot_ex/len(df)
    df = df.sample(frac=crop_frac, random_state=random_state)
    train_df = df[:int(split_frac*len(df))]
    print(train_df.head())
    test_df = df[int(split_frac*len(df)):]
    print(test_df.head())
    X_train_temp = train_df.loc[:, [i for i in range(num_ft)]]
    y_train_temp = train_df.loc[:, [num_ft]]
    X_test_temp = test_df.loc[:, [i for i in range(num_ft)]]
    y_test_temp = test_df.loc[:, [num_ft]]

    train_X = np.array(X_train_temp.values)
    train_y = np.array(y_train_temp.values)
    test_X = np.array(X_test_temp.values)
    test_y = np.array(y_test_temp.values)
    
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = load_data(file_name, split_frac, tot_ex)

# %%
"""
## Utility functions
"""

# %%
def get_folds_idx(N, nFolds, seed=42):
    """
    Randomly permute [0,N] and extract indices for each fold
    """
    np.random.seed(seed)
    rnd_idx = np.random.permutation(N)
    N_fold = N//nFolds
    indices = []
    for i in range(nFolds):
        start = i*N_fold
        end = min([(i+1)*N_fold, N])
        # if (N<end):
        #     end = N
        indices.append(rnd_idx[start:end])
    return indices


def convToList(y, out_dim):
    assert(int(y)==y)
    tmp = np.zeros(out_dim, dtype=np.int32)
    tmp[int(y)] = 1
    return tmp

def showImage(img, label):
    picAr = np.array(img, dtype='float')
    roughSd = int(math.sqrt(img.size))
    pic = picAr.reshape((roughSd, roughSd)).T
    plt.imshow(pic) #cmap='grey'
    lb = str(label)
    plt.title('label for this image is',lb)
    plt.show()

    
def act_fn(typ, Z):
    """
    Arguments:
    typ -- sigmoid/RELU
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z)/RELU(z)/tanh(Z), same shape as Z
    """
    if (typ.lower()=='sigmoid'):
        a = 1/(1+np.exp(-Z))
    elif (typ.lower()=='relu'):
        a = np.maximum(0,Z)
    elif (typ.lower()=='tanh'):
        a= np.tanh(Z)
        
    assert(a.shape == Z.shape)
    return a
    
def back_fn(typ, Z):
    if (typ.lower()=='relu'):
        #dZ = np.array(Z, copy=True)
        # When z <= 0, set dz to 0. 
        #dZ[Z <= 0] = 0
        dZ = np.array(Z>=0).astype('int')
    elif (typ.lower()=='sigmoid'):
        dZ = np.exp(-Z)/(1+np.exp(-Z))**2
    elif (typ.lower()=='tanh'):
        dZ = 1-np.tanh(Z)**2
    
    assert(dZ.shape==Z.shape)
    return dZ

    
def forCost(typ, logits, yt):
    y = convToList(yt, logits.shape[1])
    y = np.reshape(y, logits.shape)
    if (typ.lower()=='mse'):
        delt = np.power(logits-y,2)
        ret= np.mean(delt)
        #ret/=y.shape
    elif (typ.lower()=='sse'):
        delt = np.power(logits-y,2)
        ret= np.sum(delt)
        ret/=2
    elif (typ.lower()=='cross'):
        tmp = np.multiply(y, logits) + np.multiply((1-y),(1-logits))
        ret = -np.sum(tmp)
    else:
        print("Lmao ded, gonna get an error")
    return ret


def backCost(typ, logits, yt):
    y = convToList(yt, logits.shape[1])
    y = np.reshape(y, logits.shape)
    if (typ.lower()=='mse'):
        delt=(logits-y)
        ret = 2*delt/y.size
    elif (typ.lower()=='sse'):
        ret = logits-y
    elif (typ.lower()=='cross'):
        ret = -np.divide(y, logits) + np.divide(1-y, 1-logits)
        assert(ret.shape==y.shape)
    else:
        print("Lmao ded, gonna get an error")
    return ret


def update_lr(eta0, iteration):
    return eta0/((iteration+1)**0.5)

def normalize(X):
    return X/255

def predict(network, xin):
    out=np.reshape(xin, (1, -1))
    for layer in network:
        out = layer.forward(out)
    return out

def accuracy(network, test_X, test_y):
    corr = 0
    for (x,yt) in zip(test_X, test_y):
        logits = predict(network, x)
        y = convToList(yt, logits.shape[1])
        y = np.reshape(y, logits.shape)
        if (np.argmax(y) == np.argmax(logits)):
            corr+=1
    corr/=len(test_y)
    return 100*corr

# %%
"""
## Class (Cuz I'm fancy)
"""

# %%
class Layers:
    # def __init__(self, input_shape=None, input_dim=None, output_dim=None, act_fn=None, typ=None):
        
    def __init__(self, input_shape=None, input_dim=None, output_dim=None, act_fnc=None, typ=None):
        self.typ=typ
        if(typ.lower()=='softmax'):
            self.input_dim = input_dim
        elif(typ.lower()=='activation'):
            self.act_fun = act_fnc
        elif(typ.lower()=='fc'): #FC
            self.input_dim = input_dim
            self.output_dim = output_dim
            ssz = np.sqrt((input_dim)/2)
            self.weights = np.random.randn(input_dim, output_dim)/ssz
            self.bias = np.random.randn(1, output_dim)/ssz
        else:
            print("Oops! Wrong layer type. Gonna go die")

    def forward(self, xin):
        if(self.typ.lower()=='softmax'):
            self.inputSoft = xin
            #exps = np.exp(xin - xin.max())
            exps = np.exp(xin)
            self.outSoft = exps/np.sum(exps)
            return self.outSoft
        elif(self.typ.lower()=='activation'):
            self.inputAct = xin
            return act_fn(self.act_fun, xin)
        elif(self.typ.lower()=='fc'): #FC
            self.inputFC = xin
            return np.dot(xin, self.weights)+self.bias

    def backward(self, out_err, lr):
        if(self.typ.lower()=='softmax'):
            in_err=np.zeros(out_err.shape)
            out=np.tile(self.outSoft.T, self.input_dim)
            return self.outSoft*np.dot(out_err, np.identity(self.input_dim)-out) ###Can have problems
            #tmp = self.inputSoft
            #exps = np.exp(tmp-tmp.max())
            #tmp2= exps/np.sum(exps)*(1-exps/np.sum(exps))
            #return out_err*tmp2
        elif(self.typ.lower()=='activation'):
            return out_err*back_fn(self.act_fun, self.inputAct)
        elif(self.typ.lower()=='fc'): #FC
            in_err = np.dot(out_err, self.weights.T)
            assert(self.inputFC.T.shape[-1] == out_err.shape[0])
            wt_err = np.dot(self.inputFC.T, out_err)
            self.weights-=lr*wt_err
            self.bias-=lr*out_err
            return in_err

    def rettyp(self):
        return self.typ

# %%
"""
## Create Network
"""

# %%
def create_net(typ, num_nodes, activation_fn, tm):
    network = []
    for idx, ll in enumerate(typ):
        if (ll=='fc'):
            assert(idx%2==0)
            network.append(Layers(typ=ll, input_dim = num_nodes[idx//2], output_dim = num_nodes[(idx//2)+1]))
            if (tm==0):
                print('Layer type: %s \nInput Dimension: %d \t Output Dimension: %d'%(ll.title(),num_nodes[idx//2],num_nodes[(idx//2)+1]))
        elif (ll=='activation'):
            assert((idx-1)%2==0)
            network.append(Layers(typ=ll, act_fnc=activation_fn[(idx-1)//2]))
            if (tm==0):
                print('Layer type: %s \nActivation Function: %s'%(ll.title(), activation_fn[(idx-1)//2].title()))
        elif (ll=='softmax'):
            network.append(Layers(typ=ll, input_dim=num_nodes[-1]))
            if (tm==0):
                print('Layer type: %s \nInput Dimension: %d'%(ll.title(), num_nodes[-1]))
        else:
            print("You did something wrong there homie. Try again")
    
    return network

# %%
"""
## Training Function
"""

# %%
def train(network, Xtrain, ytrain, epochs=50, initlr=0.1, cost_fn='mse', 
          early_stop=False, batch_size=30, patience=2, thresh=1e-4, chng_lr=True):
    
    error = []
    checSz=0
    lr = initlr
    for epoch in range(epochs):
        checSz=epoch
        if (chng_lr==True):
            lr = update_lr(initlr, epoch)
        err = 0
        for batch in range(0, Xtrain.shape[0], batch_size):
            X_batch,y_batch= (Xtrain[batch:batch+batch_size], ytrain[batch:batch+batch_size])
            
            for (x_b,y_b) in zip(X_batch, y_batch):
                out = np.reshape(x_b, (1, -1))
                for layer in network:
                    out=layer.forward(out)

                err+=forCost(cost_fn, out, y_b)
                #print(err)
                rev_err=backCost(cost_fn, out, y_b)

                for layerIdx in range(len(network)):
                    layer = network[-1-layerIdx]
                    rev_err = layer.backward(rev_err, lr)

        err=err/len(Xtrain)
        #print('%d/%d, Error=%f' % (epoch+1, epochs, err))
        error.append(err)
        flag=False
        if (early_stop==True):
            patience=patience+1
            if(len(error)>patience):
                lastLoss = error[-1]
                for i in range(patience):
                    if (abs(error[-i-2]-lastLoss)<thresh):
                        flag=True
                        lastLoss=error[-i-2]
                    else:
                        break
                        
        if (flag==True):
            break
        
    assert(len(error)==checSz+1)
    if (checSz<epochs-1):
        print('Early stopping at %dth epoch'%(checSz+1))
    return error

# %%
"""
## User Stuff (Inputs)
"""

# %%
input_shape = (28,28) #I set this depending upon picture
prod = 1
for i in input_shape:
    prod=prod*i
num_nodes = [prod, 256, 64, 10] #List of nodes in each layer 

activation_fn= ['tanh', 'relu'] #List of activation functions for each layer
assert(len(activation_fn) == len(num_nodes)-2) #Because last will be softmax

typ= ['fc', 'activation', 'fc', 'activation', 'fc', 'softmax'] #I set it mp. len = 2*len(act_fn)-2
# ALWAYS of the type fc,act,fc,act,...,fc,softmax
# DO NOT ADD SOFTMAX ANYWHERE ELSE
assert(len(typ) == 2*len(num_nodes)-2)

#### TRAINING RELATED HYPERPARAMETERS
folds = 5
EPOCHS = 20
initial_learning_rate = 0.1
chng_lr=True
cost_fn = 'sse'
early_stop = True
batch_size=30
patience=2
thresh=1e-4

# %%
"""
# MEGA PIPELINE
"""

# %%
diff_epochs = [1, 10, 20, 50]
diff_lr = [0.001, 0.01, 0.1, 1]
diff_costs = ['mse', 'sse']
diff_chng_lr= [True, False]
diff_early_stop = [True, False]
diff_batch_size= [1, 10, 20, 50, 100]

N = len(train_X)
assert(N==len(train_y))
idx_all = np.arange(0, N)
idx_folds = get_folds_idx(N, folds, seed=random_state) # list of list of fold indices

tot_mods = 0
for EPOCHS in diff_epochs:
    for initial_learning_rate in diff_lr:
        for cost_fn in diff_costs:
            for chng_lr in diff_chng_lr:
                for early_stop in diff_early_stop:
                    for batch_size in diff_batch_size:
                        tot_mods+=1
                        print()
                        print('--------------------------------------------------')
                        print('Iteration Number:', tot_mods)
                        print('Number of Epochs: %d \nInitial Learning Rate: %f \nCost Function used: %s \nLearning Rate is being updated?: %s \nEarly Stopping Regularization being used?: %s \nBatch Size: %d'%(EPOCHS, 
                                  initial_learning_rate, cost_fn.title(), 'Yes' if chng_lr==True else 'No', 'Yes' if early_stop==True else 'No', batch_size))

                        train_acc = np.array([])
                        dev_acc = np.array([])
                        test_acc = np.array([])

                        Start = time.process_time()
                        for i,indcs in enumerate(idx_folds):
                            print("Fold Number: %d/%d"%(i+1, len(idx_folds)))
                            network = create_net(typ, num_nodes, activation_fn, i)
                            idx_train = np.delete(idx_all, indcs)
                            X_train, y_train = train_X[idx_train], train_y[idx_train]
                            X_valid, y_valid = train_X[indcs], train_y[indcs]
                            if (i==0):
                                print("Training Size:", X_train.shape)
                                print("Validation Size:", X_valid.shape)
                            try:
                                error = train(network, X_train, y_train, epochs=EPOCHS, initlr = initial_learning_rate, cost_fn=cost_fn, 
                                            early_stop=early_stop, batch_size=batch_size, patience=patience, thresh=thresh, chng_lr=chng_lr)
                                print('Initial loss:%f | Final loss:%f'%(error[0], error[-1]))
                                train_acc = np.append(train_acc, accuracy(network, X_train, y_train))
                                dev_acc = np.append(dev_acc, accuracy(network, X_valid, y_valid))
                                #test_acc = np.append(test_acc, accuracy(network, test_X, test_y))
                            except:
                                print("Some issue with this iteration... Moving on to the next one!")
                        End = time.process_time()
                        print('Mean train accuracy: %f'%(np.mean(train_acc)))
                        print('Mean dev accuracy: %f'%(np.mean(dev_acc)))
                        #print('Mean test accuracy: %f'%(np.mean(test_acc)))
                        print('Total time taken in %d-folds CV on given set of hyperparameters: %f seconds'%(folds, End-Start))

# %%
"""
## Run Specific
"""

# %%
N = len(train_X)
assert(N==len(train_y))
idx_all = np.arange(0, N)
idx_folds = get_folds_idx(N, folds, seed=random_state) # list of list of fold indices


train_acc = np.array([])
dev_acc = np.array([])
test_acc = np.array([])

Start = time.process_time()
for i,indcs in enumerate(idx_folds):
    
    network = create_net(typ, num_nodes, activation_fn, i)
    idx_train = np.delete(idx_all, indcs)
    X_train, y_train = train_X[idx_train], train_y[idx_train]
    X_valid, y_valid = train_X[indcs], train_y[indcs]

    error = train(network, X_train, y_train, epochs=EPOCHS, initlr = initial_learning_rate, cost_fn=cost_fn, 
                  early_stop=early_stop, batch_size=batch_size, patience=patience, thresh=thresh, chng_lr=chng_lr)
    print('Initial loss:%f | Final loss:%f'%(error[0], error[-1]))
    train_acc = np.append(train_acc, accuracy(network, X_train, y_train))
    dev_acc = np.append(dev_acc, accuracy(network, X_valid, y_valid))
    test_acc = np.append(test_acc, accuracy(network, test_X, test_y))

End = time.process_time()
print('Mean train accuracy: %f'%(np.mean(train_acc)))
print('Mean dev accuracy: %f'%(np.mean(dev_acc)))
print('Mean test accuracy: %f'%(np.mean(test_acc)))
print('Total time taken in %d-folds CV: %f seconds'%(folds, End-Start))

# %%
"""
## Iterate over single hyperparameter
"""

# %%
diff_epochs = [1, 5, 10, 20, 50]
diff_lr = [0.00001, 0.0001, 0.001, 0.01, 0.1]
diff_lr_lin = [0.001, 0.002, 0.004, 0.005, 0.007, 0.008]
diff_batch_size= [1, 10, 20, 50, 100, 200, 500]

N = len(train_X)
assert(N==len(train_y))
idx_all = np.arange(0, N)
idx_folds = get_folds_idx(N, folds, seed=random_state) # list of list of fold indices

# %%
EPOCHS = 20
initial_learning_rate = 0.001
cost_fn = 'sse'
chng_lr = True
early_stop = True
batch_size = 1

mean_train = np.array([])
std_train = np.array([])
mean_test = np.array([])
std_test = np.array([])
time_taken = np.array([])
print()
print('--------------------------------------------------')
print("Curves for epochs")

for EPOCHS in diff_epochs:
  
    train_acc = np.array([])
    test_acc = np.array([])

    Start = time.process_time()
    for i,indcs in enumerate(idx_folds):
        network = create_net(typ, num_nodes, activation_fn, 1)
        idx_train = np.delete(idx_all, indcs)
        X_train, y_train = train_X[idx_train], train_y[idx_train]
        X_valid, y_valid = train_X[indcs], train_y[indcs]

        error = train(network, X_train, y_train, epochs=EPOCHS, initlr = initial_learning_rate, cost_fn=cost_fn, 
                  early_stop=early_stop, batch_size=batch_size, patience=patience, thresh=thresh, chng_lr=chng_lr)
        train_acc = np.append(train_acc, accuracy(network, X_train, y_train))
        test_acc = np.append(test_acc, accuracy(network, test_X, test_y))
        
    End = time.process_time()

    time_taken = np.append(time_taken, End-Start)
    mean_train=np.append(mean_train, np.mean(train_acc))
    mean_test= np.append(mean_test, np.mean(test_acc))
    std_train= np.append(std_train, np.std(train_acc))
    std_test= np.append(std_test, np.std(test_acc))
  
  

plt.errorbar(diff_epochs, mean_train, std_train, linestyle='--', marker='o', label='train')
plt.errorbar(diff_epochs, mean_test, std_test, linestyle='-', marker='^',label='test' )
plt.grid(True)
plt.legend(loc='best')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.title('Accuracy Vs Number of epochs')
plt.show()

plt.plot(diff_epochs, time_taken)
plt.title('Time vs Number of epochs')
plt.ylabel('Time')
plt.xlabel('Epochs')
plt.show()

# %%
"""
## Representations
"""

# %%
import random

EPOCHS = 30
initial_learning_rate = 0.1
cost_fn = 'sse'
chng_lr = True
early_stop = True
batch_size = 30

N = len(train_X)
assert(N==len(train_y))
network = create_net(typ, num_nodes, activation_fn, 0)
print("Training Size:", train_X.shape)

error = train(network, train_X, train_y, epochs=EPOCHS, initlr = initial_learning_rate, cost_fn=cost_fn, 
            early_stop=early_stop, batch_size=batch_size, patience=patience, thresh=thresh, chng_lr=chng_lr)
print('Initial loss:%f | Final loss:%f'%(error[0], error[-1]))


som = []
for i, y in enumerate(train_y):
    if (random.uniform(0,1)<0.01):
        out = np.reshape(train_X[i], (1, -1))
        for layer in network:
            out = layer.forward(out)
            if(layer.rettyp()=='softmax'):
                som.append(layer.inputSoft)
            elif(layer.rettyp()=='fc'):
                if (layer.inputFC.shape!=train_X[i].shape):
                    som.append(layer.inputFC)

# %%
def showImage(img):
    plt.figure(figsize=(1,1))
    picAr = img
    roughSd = int(math.sqrt(img.size))
    #assert(roughSd==28)
    pp = roughSd**2
    picAr = picAr[:pp]
    # if (roughSd==)
    # print()
    pic = picAr.reshape((roughSd, roughSd))
    plt.imshow(pic) #cmap='grey'
    plt.show()

# %%
for i in range(0, len(som),4):
    if (random.uniform(0,1)<0.4):
        showImage(som[i])
        showImage(som[i+1])
        showImage(som[i+2])
        #showImage(som[i+3])