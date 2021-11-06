#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from time import time


# In[15]:


class Layer:
    def __init__(self, inp_len, nodes, type):    
        self.wts = np.random.randn(inp_len, nodes) / 100
        self.biases = np.zeros((1,nodes))
        self.type = type 
        
    def act_fn(self, x, t):
        if t==1 :
            return 1/(1+np.exp(-x))
        if t==2:
            return x * (x > 0.1)
                       
       
    def act_fn_prime(self,x,t) :
        if t==1 :
            return self.act_fn(x,t)*(1-self.act_fn(x,t)) 
        if t==2:
            return 1. * x>0
        
    def fprop(self,inp):
        self.last_inp = inp
        inp_len, nodes = self.wts.shape
        #print(inp.shape,self.wts.shape)
        y = np.dot(inp, self.wts) + self.biases
        self.last_y = y
        op  = self.act_fn(y,self.type)
        return op
    
    
    def bprop(self, dL_dout, lr, lam):
        dL_dt = dL_dout*self.act_fn_prime(self.last_y,self.type)
        #print ("dL_dt is ",dL_dt.shape)
        dL_dw = np.dot(self.last_inp.T,dL_dt)
        dL_db = dL_dt
        dL_dinps = np.dot(dL_dt, self.wts.T)
        self.wts = (1- lam*lr) *self.wts + lr *dL_dw
        self.biases = (1 - lam*lr) *self.biases + lr *dL_db
        
        return dL_dinps


# In[16]:


def accuracy(ref,b):
    a = np.array(ref)
    for i in range(len(a[0])):
        t = a[0][i]
		#print a
		#print t
        if t>0.5:
            a[0][i]= 1
        else:
            a[0][i] = 0
    if np.array_equal(b[0], a[0]):
        return 1
    else :
        return 0


# In[17]:


def calAccuracy(show):
    p = 0
    q = len(test_data)
    for i in range(q) :
        im = test_data[i]
        im = im.reshape(1,m)
        label = test_label[i]
        lab = np.zeros((1,10))
        lab[0][label] = 1
        lab = lab.reshape(1,n)
        _,l,a = fprop(im,label)
        p += a
        if (a==0 and show) :
            showImage(im , label)
		#print "p value is ",p
    return (p/q)*100


# In[18]:


def fprop(data_point,label):
    out = layers[0].fprop(data_point)
    #print(l)
    for i in range(s-1) :
        #print(i)
        out = layers[i+1].fprop(out)

    lab = np.zeros((1,10))
    lab[0][label] = 1
    #print (lab)
    loss = lab - out
    acr = accuracy(out,lab)
    return out,loss,acr


# In[19]:


def train(im, label, lr, loss, batch_size):
    out, l,acr = fprop(im, label)
    loss = (loss+l)/batch_size
    #print(loss)
    grad = layers[s-1].bprop(loss,lr,lam)
    for i in range(s-2,-1,-1) :
        grad = layers[i].bprop(grad,lr,lam)
    
    return loss,acr


# In[20]:


def showImage(image,label):
    fim = np.array(image, dtype='float')
    l = int(math.sqrt(image.size))
    pix = fim.reshape((l,l)).T
    plt.imshow(pix, cmap='gray')
    #lb = str(label)
    #plt.title('label for this image is',lb)
    print (label)
    plt.show()


# In[21]:


df = pd.read_csv("http://web.iitd.ac.in/~sumeet/A3/2016EE10459.csv",nrows = 3000)

data = np.array(df.values)
size = len(data) ;
le = int((len(data)*9)/10)

train_data = data[:le,:784]/255 ;
train_label = data[:le,784] ;

test_data = data[le:,:784]/255 ;
test_label = data[le:,784] ;

#df = pd.read_csv("http://web.iitd.ac.in/~sumeet/A2/2016EE10459.csv",nrows = 3000)
#
#data = np.array(df.values)
#size = len(data) ;
#le = int((len(data)*8)/10)
#
#train_data = (data[:le,:25]+7)/16 ;
#train_label = data[:le,25].astype(int) ;
#
#test_data = (data[le:,:25]+7)/16 ;
#test_label = data[le:,25].astype(int) ;

# In[22]:


m = len(train_data[0])
n = 10
node = [m,36,n]
af = [1,1]
#out_layer = Layer(m,n)
lr = 1
batch_size = 1
conv_crit = 99
epoch_size = 50
lam = 0


# In[23]:


layers = []
s = len(af) 
for i in range(s):
    a = node[i]
    b = node[i+1]
    c = af[i]
    layers.append(Layer(a,b,c))
accur = np.zeros((0,4))    


# In[24]:


sc = 0
epoch = 0
#for ep in range(epoch_size):
while (sc<conv_crit and epoch<epoch_size ) :
    print('--- Epoch %d ---' % (epoch + 1))

    loss = 0
    num_correct = 0
    t1 = time()
    for i in range(len(train_data)):
        im = train_data[i]
        im = im.reshape(1,m)
        label = train_label[i]
        #label = label.reshape(1,n)
        #print(im.shape , label.shape)
        if (i>0 and (i % batch_size ==0)):
            l,a = train(im,label,lr,loss,batch_size)
            loss = 0
            num_correct += a
            #print("loss is ",l)
            #plt.scatter(epoch*(len(train_data))+i,num_correct*100/i)
        else:
            #print(im.shape)
            u,l,a = fprop(im,label)
            loss+=l
            num_correct +=a
            #print("loss is ",loss/batch_size)
    t2 = time()
    sc = num_correct*100/len(train_data)
    temp = np.zeros((1,4))
    temp[0][0] = epoch+1
    temp[0][1] = sc
    temp[0][2] = calAccuracy(False)
    temp[0][3] = t2-t1
    accur = np.append(accur,temp,axis=0)
    print("accuracy is ",sc)
    epoch += 1
    
lbl1 =  'train accuracy' 
lbl2 =  'test accuracy'
plt.plot(accur[0:,0],accur[0:,1],label = lbl1)
plt.plot(accur[0:,0],accur[0:,2],label = lbl2)
plt.title('accuracy vs epoch for FCNN')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

plt.plot(accur[0:,0],accur[0:,3])
plt.title('time vs epoch for FCNN')
plt.ylabel('time in second')
plt.xlabel('epoch')
plt.legend()
plt.show()


# In[25]:


print ("test accuracy is ",calAccuracy(False))


# In[13]:
img = train_data[1]
lbl = train_label[1]
showImage(img,lbl)
out = layers[0].fprop(img)
showImage(out,lbl)
    #print(l)
for i in range(s-1) :
    #print(i)
    out = layers[i+1].fprop(out)
    showImage(out,lbl)