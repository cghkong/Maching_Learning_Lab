import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from scipy.stats import multivariate_normal
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

"""
产生k个高斯分布，每个分布有n个数据
得到的是带有真实类别标签的X，以便于绘图
"""
def generate_data(k, n, dim, mean, sigma):
    data = np.zeros((n * k, dim + 1))
    for i in range(k):
        data[i * n : (i + 1) * n, :dim] = np.random.multivariate_normal(mean[i], sigma[i], size=n)
        data[i * n : (i + 1) * n, dim : dim + 1] = i
    return data

"""
标准iris数据集(鸢尾花)
"""
def uci_iris():
    data_set = load_iris()
    Xiris = data_set["data"]
    Yiris = data_set["target"].reshape(Xiris.shape[0],1)
    data = np.append(Xiris,Yiris,axis=1)
    return data


"""
实现K-means，计算分类结果和中心
"""
def kmeans(X, k, epsilon=1e-5):
    center = np.zeros((k, X.shape[1] - 1))
    # 随机初始化聚类中心
    for i in range(k):
        center[i, :] = X[np.random.randint(0, high=X.shape[0]), :-1]
    iter = 0
    while True:
        iter += 1
        distance = np.zeros(k)
        # 根据中心重新给每个点贴分类标签
        for i in range(X.shape[0]):
            for j in range(k):
                distance[j] = np.linalg.norm(X[i, :-1] - center[j, :])
            X[i, -1] = np.argmin(distance)    #选取最近的中心
        # 更新中心
        new_center = np.zeros((k, X.shape[1] - 1))
        count = np.zeros(k)
        for i in range(X.shape[0]):
            new_center[int(X[i, -1]), :] += X[i, :-1]  # 对每个类的所有点坐标求和
            count[int(X[i, -1])] += 1
        for i in range(k):
            new_center[i, :] = new_center[i, :] / count[i]  # 对每个类的所有点坐标求平均值
        if np.linalg.norm(new_center - center) < epsilon:  # 用差值的二范数表示精度
            break
        else:
            center = new_center
    print("聚类迭代次数：",iter)
    return X, center, iter


'''
EM算法的E步骤，求样本后验概率
'''
def EM_E(x, mean, sigma, pi):
    k = mean.shape[0]
    pz = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        pi_pdf_sum = 0
        pi_pdf = np.zeros(k)
        for j in range(k):
            pi_pdf[j] = pi[j] * multivariate_normal.pdf(x[i], mean=mean[j], cov=sigma[j])
            pi_pdf_sum += pi_pdf[j]
        for j in range(k):
            pz[i, j] = pi_pdf[j] / pi_pdf_sum
    return pz

'''
EM算法的M步骤，更新参数
'''
def EM_M(x, mean, gamma_z):
    k = mean.shape[0]
    n = x.shape[0]
    dimension = x.shape[1]
    mean_new = np.zeros(mean.shape)
    sigma_new = np.zeros((k, dimension, dimension))
    pi_new = np.zeros(k)
    for j in range(k):
        n_j = np.sum(gamma_z[:, j])
        pi_new[j] = n_j / n  # 计算新的pi
        gamma = gamma_z[:, j]
        gamma = gamma.reshape(n, 1)
        mean_new[j, :] = (gamma.T @ x) / n_j  # 计算新的mean
        sigma_new[j] = ((x - mean[j]).T @ np.multiply((x - mean[j]), gamma)) / n_j  # 计算新的sigma
    return mean_new, sigma_new, pi_new


"""
计算极大似然估计（对数）
"""
def likelihood(x, mean, sigma, pi_list):
    loss = 0
    for i in range(x.shape[0]):
        pi_sum = 0
        for j in range(mean.shape[0]):
            pi_sum += pi_list[j] * multivariate_normal.pdf(x[j], mean=mean[j], cov=sigma[j])
        loss += np.log(pi_sum)
    return loss

'''
GMM算法
'''
def gmm(X, k, epsilon=1e-5):
    x = X[:, :-1]
    pi_list = np.ones(k) * (1.0 / k)
    sigma = np.array([0.1 * np.eye(x.shape[1])] * k)
    # 随机选第1个初始点，依次选择与当前mu中样本点距离最大的点作为初始簇中心点
    mean = [x[np.random.randint(0, k) + 1]]
    for times in range(k - 1):
        temp_ans = []
        for i in range(x.shape[0]):
            temp_ans.append(
                np.sum([np.linalg.norm(x[i] - mean[j]) for j in range(len(mean))])
            )
        mean.append(x[np.argmax(temp_ans)])
    mean = np.array(mean)

    old_l = likelihood(x, mean, sigma, pi_list)
    iter = 0
    log_l = pd.DataFrame(columns=("iter", "log likelihood"))
    while True:
        gamma_z = EM_E(x, mean, sigma, pi_list)
        mean, sigma, pi_list = EM_M(x, mean, gamma_z)
        new_l = likelihood(x, mean, sigma, pi_list)
        if iter % 10 == 0:
            log_l = log_l.append(
                [{"iter": iter, "log likelihood": old_l}],
                ignore_index=True,
            )
        if old_l < new_l and (new_l - old_l) < epsilon:
            break
        old_l = new_l
        iter += 1
    # 计算标签
    for i in range(X.shape[0]):
        X[i, -1] = np.argmax(gamma_z[i, :])
    return X, iter, log_l

"""
计算准确率
"""
def compute_accuracy(real_lable, class_lable, k):
    classpatten = list(permutations(range(k), k))
    counts = np.zeros(len(classpatten))
    for i in range(len(classpatten)):
        for j in range(real_lable.shape[0]):
            if int(real_lable[j]) == classpatten[i][int(class_lable[j])]:
                counts[i] += 1
    return np.max(counts) / real_lable.shape[0]


'''
综合聚类结果
'''
def kmeans_result(X, k, realclass, epsilon):
    Xk, center, iter = kmeans(X, k, epsilon=epsilon)
    print(center)
    accuracy = compute_accuracy(realclass, Xk[:, -1], k)
    show(X,center,title="kmeans    iter="+ str(iter)+ "    accuracy="+ str(accuracy))

'''
综合GMM算法的结果
'''
def gmm_result(X, k, real_lable, epsilon):
    Xg, iter, log_l = gmm(X, k, epsilon=epsilon)
    accuracy = compute_accuracy(real_lable, Xg[:, -1], k)
    show(X,title="gmm    iter="+ str(iter)+ "    accuracy="+ str(accuracy))

'''
可视化展示
'''
def show(X, center=None, title=None):
    plt.style.use("fast")
    plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], marker=".", s=40, cmap="rainbow")
    if not center is None:
        plt.scatter(center[:, 0], center[:, 1], c="r", marker="x", s=250)
    if not title is None:
        plt.title(title)
    plt.show()


if __name__ =='__main__':
    k=4
    d=2
    n=30
    #n=[30,40,20,50]
    mean = [[2,3],[6,10],[7,1],[2,8]]
    sigma = []
    sigma1 = [[1,2],[2,1.5]]
    sigma2 = [[1.5,1],[1,2]]
    sigma3 = [[3,1.5],[1.5,2.5]]
    sigma4 = [[1,4],[4,3]]
    '''
    sigma1 = [[1,0],[0,1.5]]
    sigma2 = [[1.5,0],[0,2]]
    sigma3 = [[3,0],[0,2.5]]
    sigma4 = [[1,0.5],[0.5,1.5]]
    '''
    sigma.append(sigma1)
    sigma.append(sigma2)
    sigma.append(sigma3)
    sigma.append(sigma4)
    X1 = generate_data(k,n,d,mean,sigma)
    epsilon=0.00000001
    realclass = X1.copy()
    gmm_result(X1, k, realclass[:,-1], epsilon)
    #kmeans_result(X1, k, realclass[:,-1], epsilon)
    #X2 = uci_iris()
    #real_iris = X2.copy()

    #Xk1, centerk, itk = kmeans(X2,3,epsilon)
    #print(compute_accuracy(real_iris[:,-1],Xk1[:,-1],3))

    #xg1,logl,iterg = gmm(X2,3,epsilon)
    #print(compute_accuracy(real_iris[:,-1],xg1[:,-1],3))
