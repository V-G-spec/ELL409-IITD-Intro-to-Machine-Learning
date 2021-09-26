import argparse  
from code_part1 import foo
import numpy as np
from numpy.linalg import inv
# import pandas as pd
# import matplotlib.pyplot as plt




def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  #pinv or gd
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="1A ", type=str, help = "Read content from the file")
    return parser.parse_args()
    
def read_data_noob(file_path, method, train_size=None):
    X = np.array([])
    y = np.array([])
    file_data = np.genfromtxt(file_path, dtype=float, delimiter=",")
#     print(file_data[:5])
    for row in file_data:
        X = np.append(X, row[0])
        y = np.append(y, row[1])
    np.random.shuffle(X)
    np.random.shuffle(y)
#     print(X[:5], y[:5])
    return X, y

# def read_data(file_path, method, train_size=None):
#     df = pd.read_csv(file_path, header=None)
#     df = df.sample(n = len(df))
    
#     X = np.array(df.values)[:, 0:1]
#     y = np.array(df.values)[:, 1:]
    
#     if train_size==None:
#         train_size = int(0.85*len(df))
#     train_df = df[:train_size]
#     test_df = df[train_size:]
#     # Convert everything to a numpy array, as it makes my life easy. Works pretty much like MATLAB after that
#     train_x = np.array(train_df.values)[:, 0:1]
#     train_y = np.array(train_df.values)[:, 1:]
#     test_x = np.array(test_df.values)[:, 0:1]
#     test_y = np.array(test_df.values)[:, 1:2]
    
#     if (method=="pinv"):
#         return X, y
#     else:
#         return train_x, train_y, test_x, test_y


    
def pinvtry(file_path, method, polnyomial, result_dir, lamb):
    X, y = read_data_noob(file_path = file_path, method = method)
    N = len(X)
    temp_X = X
    X_new = X
#     E_rms = []

    M = np.arange(2, polynomial+1)
    for m in M:
        temp_X = temp_X*X
        X_new = np.c_[X_new, temp_X]
    #y_hat = X_fin.dot(theta_best)
    #temp_val = sum((y_hat-y)**2)
    #temp_val /= len(y)
    #E_rms.append(temp_val)
    
    X_fin = np.c_[np.ones((N, 1)), X_new]
    #wt_best = np.linalg.pinv(X_fin).dot(y)
    wt_best = np.linalg.inv((lamb*np.identity(polynomial+1)) + X_fin.T.dot(X_fin)).dot(X_fin.T).dot(y)
    
    y_hat = X_fin.dot(wt_best)
    err = sum((y_hat-y)**2)
    #err/=N
    #disperr(err)
    dispwt(wt_best)
    return;


        
def gdtry(file_path, method, polynomial, batch_size, lamb, result_dir):
    
    lr = 1e-3
    wt = np.random.randn(polynomial+1,1)
    M = np.arange(2, polynomial+1)
    
    X, y = read_data_noob(file_path = file_path, method = "pinv")
    N  = len(X)
    
    temp_X = X
    X_new = X
    for m in M:
        temp_X = temp_X*X
        X_new = np.c_[X_new, temp_X]
    X_fin = np.c_[np.ones((len(X_new), 1)), X_new]
    
    for i in range(0, N, batch_size):
        X_i = X_fin[i:i+batch_size]
        y_i = y[i:i+batch_size]
        grad = ((1/batch_size) * X_i.T.dot(X_i.dot(wt) - y_i)) + (lamb/batch_size)*wt
        wt = wt - lr*grad
    
    y_hat = X_fin.dot(wt)
    err = sum((y_hat-y)**2)
    #err/=N
    disperr(err)
    dispwt(wt)
    return;



def dispwt(argg):
    print("weights=%s"%(argg.flatten()))
    return;

def disperr(argg):
    print("Error=%s"%(argg.flatten()))
    return;
    
    
    
if __name__ == '__main__':
    args = setup()
    args = vars(args)
    method = args["method"]
    batch_size = args["batch_size"]
    lamb = args["lamb"]
    polynomial = int(args["polynomial"])
    result_dir = args["result_dir"]
    X = args["X"]
    if (method=="pinv"):
        pinvtry(X, method, polynomial, result_dir, lamb)
    elif (method=="gd"):
        gdtry(X, method, polynomial, batch_size, lamb, result_dir)
    else:
        print("Not an option. Exiting code...")
#         foo.demo(args)