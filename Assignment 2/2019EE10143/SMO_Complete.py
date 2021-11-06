# %%
import pandas as pd
# !pip install cvxopt
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
import random
import time

# %%
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
# warnings.resetwarnings()
warnings.filterwarnings("ignore")

# %%
def getKer(X1, X2, which, gamma, coef0=0., degree=3):
    H = np.dot(X1, X2.T)
    if (which=='linear'):
        H = H
    elif (which=='poly'):
        H = (gamma*H+coef0)**degree
    elif (which=='sigmoid'):
        H = np.tanh(gamma*H+coef0)
    elif (which=='rbf'):
        H1 = np.diag(X1.dot(X1.T)).reshape(-1, 1)*np.ones((1, X2.shape[0]))
        H2 = np.diag(X2.dot(X2.T)).reshape(1, -1)*np.ones((X1.shape[0],1))
        H = 2*H-H1-H2
        H = np.exp(gamma*H)
    return H
  

    
def takestep(i1, i2, alphas, y, X, b, which, gamma, C, wt, eps=1e-5):
    if (i1==i2):
        return 0, alphas, b, wt
    # alphas[i1]
    y1, y2 = y[i1], y[i2]
    fxi1 = ((alphas*y).T@(getKer(X,X[i1,:], which, gamma)))*1. + b
    E1 = fxi1 - y1
    fxi2 = ((alphas*y).T@(getKer(X,X[i2,:], which, gamma)))*1. + b
    E2 = fxi2 - y2
    s = y1*y2
    if (y1!=y2):
        L = max(0, alphas[i2] - alphas[i1])
        H = min(C, C + alphas[i2] - alphas[i1])
    else:
        L = max(0, alphas[i2] + alphas[i1] - C)
        H = min(C, alphas[i2] + alphas[i1])
    if (L==H):
        return 0, alphas, b, wt
    k11 = getKer(X[i1,:],X[i1,:], which, gamma)
    k12 = getKer(X[i1,:],X[i2,:], which, gamma)
    k22 = getKer(X[i2,:],X[i2,:], which, gamma)
    eta = k11+k22-2*k12
    if(eta>0):
        a2 = alphas[i2] + y2*(E1-E2)/eta
        if (a2<L):
            a2 = L
        elif (a2>H):
            a2=H
    else:
        f1 = y1*(E1+b) - alphas[i1]*k11 - s*alphas[i2]*k12
        f2 = y2*(E2+b) - s*alphas[i1]*k12 - alphas[i2]*k22
        L1 = alphas[i1] + s*(alphas[i2]-L)
        H1 = alphas[i1] + s*(alphas[i2]-H)
        psiL = L1*f1 + L*f2 + 0.5*L1*L1*k11 + 0.5*L*L*k22 + s*L*L1*k12
        psiH = H1*f1 + H*f2 + 0.5*H1*H1*k11 + 0.5*H*H*k22 + s*H*H1*k12
        Lobj = psiL
        Hobj = psiH
        if (Lobj<Hobj-eps):
            a2=L
        elif(Lobj>Hobj+eps):
            a2=H
        else:
            a2 = alphas[i2]
    if (abs(a2-alphas[i2])<eps):
        return 0, alphas, b, wt
    a1 = alphas[i1] + s*(alphas[i2]-a2)
    b1 = b - (E1 + y1*(a1-alphas[i1])*k11 + y2*(a2-alphas[i2])*k12)
    b2 = b - (E2 + y1*(a1-alphas[i1])*k12 + y2*(a2-alphas[i2])*k22)
#     if (0<a1) and (a1<C):
#         b = b1
#     elif (0 < a2) and (a2<C):
#         b = b2
#     else:
#         
    b = (b1 + b2)/2.
    if (which=='linear'):
        wt += y1*(a1-alphas[i1])*X[i1,:] + y2*(a2-alphas[i2])*X[i2,:]
    alphas[i1] = a1
    alphas[i2] = a2
    return 1, alphas, b, wt
    
def examineEg(i2, y, alphas, X, b, tol, C, wt, gamma):
    y2 = y[i2]
    alph2 = alphas[i2]
    fxi2 = ((alphas*y).T@(getKer(X,X[i2,:], which, gamma)))*1. + b
    # print(fxi2.shape)
    E2 = fxi2 - y2
#     print(E2.shape)
    r2 = E2*y2
#     print(r2.shape, tol, alph2.shape, C)
    flag1 = (r2<-tol) and (alph2<C)
    flag2 = (r2>tol) and (alph2>0)
    if (flag1 or flag2):
        # noncz = [x if (x!=0 and x!=C) for x in alphas]
        noncz = np.where((alphas!=0) & (alphas!=C))[0]
        fx = ((alphas*y).T@(getKer(X,X, which, gamma)))*1. + b
#         print(fx.shape)
        Err = fx.T-y
#         print(Err)
#         Err=Err.T
#         print(Err.shape)
        ############################### Probable location of error #######################
        if (len(noncz)>1):
            if (Err[i2]>0):
                i1 = np.argmin(np.array(Err))
            elif (Err[i2]<=0):
                i1 = np.argmax(np.array(Err))
            flag, alphas, b, wt  = takestep(i1, i2, alphas, y, X, b, which, gamma, C, wt)
            if (flag==1):
                return 1, alphas, b, wt
        random.shuffle(noncz)
        for i1 in (noncz):
            flag, alphas, b, wt  = takestep(i1, i2, alphas, y, X, b, which, gamma, C, wt)
            if (flag==1):
                return 1, alphas, b, wt
        alphastemp = alphas.copy()
        random.shuffle(alphastemp)
        for i1 in range(len(alphastemp)):
            flag, alphas, b, wt  = takestep(i1, i2, alphas, y, X, b, which, gamma, C, wt)
            if (flag==1):
                return 1, alphas, b, wt
    return 0, alphas, b, wt

def routine(X,y,C,tol,gamma,which):
    numChanged = 0
    examineAll =1
    m,n = X.shape
    alphas = np.zeros((m,1))
    wt = np.zeros((1, n))
    b = 0
    
    while(numChanged>0 or examineAll==1):
        numChanged = 0
        if (examineAll):
            for i in range(m):
                chng, alphas, b, wt = examineEg(i, y, alphas, X, b, tol, C, wt, gamma)
                numChanged+=chng
        else:
            noncz = np.where((alphas!=0) & (alphas!=C))[0]
            for i in noncz:
                chng, alphas, b, wt = examineEg(i, y, alphas, X, b, tol, C, wt, gamma)
                numChanged+=chng
        if (examineAll==1):
            examineAll=0
        elif (numChanged==0):
            examineAll=1
    return alphas, b, wt


def predict(newX, wt, b):
    cl = (wt@(newX.T)+b>0)
    if (cl):
        return 1
    else:
        return -1

    
def getScore(X_test, y_test, alphas, which, X, y, b, gamma, threshold=1e-4):
    
    idx = np.where(alphas>threshold)[0]
    #Extract support vectors
    sX = X[idx, :]
    sy = y[idx]
    alphas = alphas[idx]
    
    ynew = np.zeros((X_test.shape[0],))
    Htemp = getKer(sX, X_test, which, gamma)
    rightcnt = 0.
    for i in range(ynew.shape[0]):
        ynew[i] = np.sum(alphas*sy*Htemp[:,i].reshape(-1,1))+b
        if (ynew[i]*y_test[i]>0):
            rightcnt+=1.
    y_pred = np.sign(ynew)
    score = rightcnt*100.0/ynew.shape[0]
    return score

# %%
file_name = '2019EE10143.csv'
df = pd.read_csv(file_name, header=None)

random_state = 69420
split_frac = 0.8
c=1
gamma=0.01
tol=0.001
max_passes=25
pairs = [(0,1), (4,6), (8,9)]
features = [10,25]
# typ = ['linear', 'sigmoid', 'poly', 'rbf']
typ = ['linear', 'sigmoid']
for (lab1, lab2) in pairs:
    for num_ft in features:
        for which in typ:
            print("#####################################################################")
            print("Labels:",lab1, ",", lab2)
            print("Number of features:", num_ft)
            print("Current kernel:", which)
            print("#####################################################################")

            #CONVERT TO USE
            df_temp = df.loc[df[25].isin([lab1, lab2])]
            #print(len(df_temp))
            df_temp.iloc[df_temp[25] == lab1, 25] = -1
            df_temp.iloc[df_temp[25] == lab2, 25] = 1
            df_temp = df_temp.sample(frac=1., random_state=random_state)
            # SPLIT IN TRAIN AND TEST
            train_df = df_temp[:int(split_frac*len(df_temp))]
            test_df = df_temp[int(split_frac*len(df_temp)):]
            # SPLIT BY FEATURES
            X_train_temp = train_df.loc[:, [i for i in range(num_ft)]]
            y_train_temp = train_df.loc[:, [25]]
            X_test_temp = test_df.loc[:, [i for i in range(num_ft)]]
            y_test_temp = test_df.loc[:, [25]]

            train_X = np.array(X_train_temp.values)
            train_y = np.array(y_train_temp.values)
            test_X = np.array(X_test_temp.values)
            test_y = np.array(y_test_temp.values)

            print ("Number of training examples:", train_X.shape, train_y.shape)
            print ("Number of test examples:", test_X.shape, test_y.shape)

            start = time.process_time()
            alphas, b, wt = routine(train_X, train_y,c,tol,gamma,which)
#             print(b,wt)
            end = time.process_time()
            if which=='linear':
                y_pred = np.zeros((train_X.shape[0]))
                rightcnt = 0.
                for idx, x in enumerate(train_X):
                    y_pred[idx] = predict(x, wt, b)
                    if (y_pred[idx]==train_y[idx]):
                        rightcnt+=1
                acc = rightcnt/train_X.shape[0]
                print("Training accuracy by SMO is:", 100*acc, "%")

                y_pred = np.zeros((test_X.shape[0]))
                rightcnt = 0.
                for idx, x in enumerate(test_X):
                    y_pred[idx] = predict(x, wt, b)
                    if (y_pred[idx]==test_y[idx]):
                        rightcnt+=1
                acc = rightcnt/test_X.shape[0]
                print("Test accuracy by SMO is:", 100*acc, "%")
            else:
                acc = getScore(train_X, train_y, alphas, which, train_X, train_y, b, gamma)
                print("Training accuracy by simplified SMO is:", acc, "%")

                acc = getScore(test_X, test_y, alphas, which, train_X, train_y, b, gamma)
                print("Test accuracy by simplified SMO is:", acc, "%")

            print("Time taken by simplified SMO for labels ({l1},{l2}) for {nf} features is {tt} seconds".format(l1=lab1, l2=lab2, nf=num_ft, tt=end-start))
            print()
            print()
            print()

# %%
