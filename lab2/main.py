import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pylab as pl
from sklearn.datasets import load_breast_cancer
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus']=False
pl.rcParams['font.sans-serif']=['SimHei']


sigmod = lambda x : 1 / (1 + np.exp(-x))


def generate_data(cov,mean1,mean2,pos_number,neg_number):
    X = np.zeros((pos_number+neg_number,2))
    X[:pos_number] = np.random.multivariate_normal(mean1,cov,pos_number)
    X[pos_number:] = np.random.multivariate_normal(mean2,cov,neg_number)
    Y = np.zeros(pos_number + neg_number)
    Y[:pos_number] = 1
    Y[pos_number:] = 0
    X = np.array(X)
    one = np.ones(pos_number + neg_number).reshape(pos_number + neg_number, 1)
    X = np.append(X, one, axis=1)
    return X,Y


def blood():
    df= pd.read_csv('data\\blood.csv')
    x = np.array(df.iloc[:,:4])
    y = np.array(df["whether he/she donated blood in March 2007"])
    one = np.ones(x.shape[0]).reshape(x.shape[0], 1)
    x = np.append(x, one, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
    return X_train,y_train,X_test,y_test

def compute_loss(w,x,y,r):
    w = np.matrix(w)
    x = np.matrix(x)
    result = 0
    for i in range(len(y)):
        result += (np.log(1+np.exp(x[i]*w)) - y[i]*(x[i] * w) + 0.5*r * w.T * w)
    return result


def compute_1_gradient(w,x,y,r):
    w = np.matrix(w)
    x = np.matrix(x)
    result = (r*w).astype(np.float64)
    for i in range(x.shape[0]):
        temp = sigmod(float(np.matrix(x[i])*w)) - y[i]
        result += (temp *(np.matrix(x[i]).T))
    result = np.matrix(result)
    return result


def compute_2_gradient(w,x,y,r):
    w = np.matrix(w)
    x = np.matrix(x)
    result = r * np.identity(w.shape[0])
    for i in range(x.shape[0]):
        t1 = float(sigmod(np.matrix(x[i])*w))
        t2 = float(sigmod(-np.matrix(x[i])*w))
        t = t1 * t2
        result += (t*(np.matrix(x[i]).T * np.matrix(x[i])))
    result = np.matrix(result)
    return result


def GD(x,y,r,w_0,rate,deta):
    pre_loss = compute_loss(w_0,x,y,r)
    #print(pre_loss)
    pre_w = w_0
    pro_w = pre_w - rate * compute_1_gradient(pre_w,x,y,r)
    pro_loss = compute_loss(pro_w,x,y,r)
    #print(pro_loss)
    k = 0
    while True:
        if pro_loss > pre_loss:
            rate = rate * 0.9
        pre_loss = pro_loss
        pre_w = pro_w
        pro_w = pre_w - rate * compute_1_gradient(pre_w,x,y,r)
        pro_loss = compute_loss(pro_w,x,y,r)
        k = k + 1
        if float(compute_1_gradient(pro_w,x,y,r).T * compute_1_gradient(pro_w,x,y,r)) < deta and float(pro_loss)>0.1:
            pro_w = pro_w + np.random.uniform(-1,1)
            rate=0.001
        if float(pro_loss)<2.5:
            break
    print(k)
    return pro_w


def Newton(w_0,x,y,r,deta):
    w = w_0
    k=0
    while np.linalg.norm(compute_1_gradient(w,x,y,r)) > deta:
        w1 = w - np.linalg.pinv(compute_2_gradient(w,x,y,r)) *compute_1_gradient(w,x,y,r)
        w = w1
        k = k+1
    print(k)
    return w


def pridict(test_x,test_y,w):
    tl = len(test_y)
    ans = 0
    for i in range(tl):
        if (float(np.matrix(test_x[i])*w))>0 & test_y[i]==1:
            ans=ans+1
        if (float(np.matrix(test_x[i])*w))<0 & test_y[i]==0:
            ans=ans+1
    acc = ans / tl
    return acc


def drawplt(w,x,y,positive,name):
    plt.scatter(x[:positive],y[:positive],color='red')
    plt.scatter(x[positive:],y[positive:],color='blue')
    x1 = np.linspace(1,9,100)
    y1 = []
    w = np.array(w.T)
    for i in range(100):
        t = -(w[0][0]*x1[i] + w[0][2])
        y1.append(t/w[0][1])
    plt.plot(x1,y1,color='purple')
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    cov = [[1,0],[0,2]]
    positive_mean = (2,3)
    negative_mean = (5,6)
    pos_number = 30
    neg_number = 20
    X,Y = generate_data(cov,positive_mean,negative_mean,pos_number,neg_number)

    X_train,y_train,X_test,y_test = blood()

    #drawplt(X[:,0],X[:,1],pos_number)
    w_0 = np.matrix(np.zeros(3))
    w_c = np.matrix(np.zeros(5))
    w_0 = w_0.T
    w_c = w_c.T
    #print(w_0)
    r = 0.1
    rate = 0.001
    r1=0
    r2=0.1
    wg1 = GD(X,Y,r1,w_0,rate,0.0001)
    #wg2 = GD(X,Y,r2,w_0,rate,0.0001)
    #w1 = Newton(w_0,X,Y,0,0.000001)
    #w11 = Newton(w_0,X,Y,r,0.000001)
    #wc = Newton(w_c,X_train,y_train,r,0.000001)
    #print(w0)
    #print(w1)
    #print(wc)
    #print(pridict(X_test,y_test,wc))
    #drawplt(w1,X[:,0],X[:,1],pos_number,"牛顿法（没有惩罚项）")
    #drawplt(w11,X[:,0],X[:,1],pos_number,"牛顿法（添加惩罚项）")
    drawplt(wg1,X[:,0],X[:,1],pos_number,"梯度下降法（没有惩罚项）")
    #drawplt(wg2,X[:,0],X[:,1],pos_number,"梯度下降法（添加惩罚项）")
    #drawplt(w0,X[:,0],X[:,1],pos_number)








