import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import random
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['axes.unicode_minus']=False
pl.rcParams['font.sans-serif']=['SimHei']

func = lambda x : np.sin(x)


#产生数据集，添加高斯噪音
def data(left,right,num):
    x = np.linspace(left,right,num)
    y0 = func(x)
    y = y0 + np.random.normal(scale=0.1, size=len(x))
    return x , y


#计算X矩阵
def compute_X_m(X,m):
    res1 = []
    for i in range(len(X)):
        res = []
        for j in range(m):
            res.append(X[i]**j)
        res1.append(res)
    result = np.matrix(res1)
    return result


#没有惩罚项的解析法
def analysis_without_penalty(X_m,T_y):
    X_m = np.matrix(X_m)
    T_y = np.matrix(T_y)
    return np.linalg.inv(X_m.T * X_m) * X_m.T * T_y


#添加惩罚项的解析法
def analysis_penalty(X_m,T_y,r):
    X_m = np.matrix(X_m)
    T_y = np.matrix(T_y)
    return np.dot(np.dot(np.linalg.inv(np.dot(X_m.T , X_m) + r * np.identity(X_m.shape[1])),X_m.T),T_y)


#计算梯度
def cal_gredient(X_m,T_y,r,pre_w):
    X_m = np.matrix(X_m)
    T_y = np.matrix(T_y)
    pre_w = np.matrix(pre_w)
    return 0.5 * ((X_m.T @ X_m) @ pre_w - (X_m.T @ T_y) + r*pre_w)

#计算损失值
def cal_loss(X_m,T_y,w,r):
    X_m = np.matrix(X_m)
    T_y = np.matrix(T_y)
    w = np.matrix(w)
    px = np.matrix(X_m @ w- T_y)
    return 0.5 * (float(px.T @ px + r* (w.T @ w)))


#梯度下降法
def GD(X_m,T_y,rate,w_0,r,deta):
    pre_loss = cal_loss(X_m,T_y,w_0,r)
    print(pre_loss)
    pre_w = w_0
    pro_w = pre_w - rate*cal_gredient(X_m,T_y,r,pre_w)
    pro_loss = cal_loss(X_m,T_y,pro_w,r)
    print(cal_gredient(X_m,T_y,r,pre_w))
    print(pro_loss)
    k = 0
    while np.abs(pro_loss - pre_loss) > deta:
        if pro_loss > pre_loss:
            rate = rate * 0.5
        pre_loss = pro_loss
        pre_w = pro_w
        pro_w = pre_w - rate*cal_gredient(X_m,T_y,r,pre_w)
        pro_loss = cal_loss(X_m,T_y,pro_w,r)
        k = k+1
    print(k)
    return pro_w


#共轭梯度法
def CG(X_m,T_y,w_0,r,deta):
    X_m = np.matrix(X_m)
    T_y = np.matrix(T_y)
    pt = np.matrix(X_m.T*X_m + r * np.identity(X_m.shape[1]))
    pre_r = np.matrix(X_m.T * T_y - pt * w_0)
    p0 = pre_r
    p1 = p0
    pro_r = pre_r
    pre_w = w_0
    pro_w = pre_w
    k = 0
    while pro_r.T * pro_r > deta:
        Ak = float(pre_r.T * pre_r) / float(p0.T * pt * p0)
        pro_w = pre_w + Ak * p0
        pre_w = pro_w
        pro_r = pre_r - Ak * pt * p0
        b1 = float(pro_r.T * pro_r) / float(pre_r.T * pre_r)
        p1 = pro_r + b1*p0
        p0 = p1
        pre_r = pro_r
        k= k+1
    print(k)
    return pro_w


#可视化绘图
def drawplt(w,X,X_m,Y,title):
    Y = Y.reshape(-1,1)
    Y1 = [float(i) for i in Y]
    x1 = np.linspace(0, 2*np.pi, 80)
    py = []
    for t in x1:
        tp = 0
        for j in range(len(w)):
            tp += float(w[j]) * (t ** j)
        py.append(tp)
    Py = [float(i) for i in py]
    plt.plot(list(x1),Py,color='red')
    #print(Y1)
    plt.scatter(list(X), Y1)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    X,Y = data(0,2*np.pi,20)
    t = 9
    X_m = compute_X_m(X,t)
    X_m = np.matrix(X_m)
    Y = np.matrix(Y)
    Y = Y.T
    #w0 = [0.25,0.26,0.005,0,0,0,0,0,0]
    #w0 = [0,0,0,0,0,0,0,0,0]
    w0 = [0,0,0,0,0,0,0,0,0]
    w0 = np.matrix(w0)
    w0 = w0.T
    rp = 0.1
    rgd = 0.001
    rcg = 0.001
    #w = analysis_without_penalty(X_m,Y)
    #w_p = analysis_penalty(X_m,Y,rp)
    #rate = 0.0000000005
    #rate = 0.00001
    #w_GD = GD(X_m,Y,rate,w0,rgd,0.000001)
    w_CG = CG(X_m,Y,w0,rcg,0.000001)

    #print(w_GD)
    #print(w_CG)
    #drawplt(w,X,X_m,Y.T,"最小二乘法拟合(没有正则项)")
    #drawplt(w_p,X,X_m,Y.T,"最小二乘法拟合（添加正则项）")
    #drawplt(w_GD,X,X_m,Y.T,"梯度下降法拟合")
    drawplt(w_CG,X,X_m,Y.T,"共轭梯度拟合")















