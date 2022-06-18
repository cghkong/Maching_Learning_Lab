
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

size = (50, 50)

""" 
进行PCA(Principal Component Analysis)
data:为原始数据
reduced_dimension:为需要降低到的维数 
"""
def pca(data, reduced_dimension):
    rows, columns = data.shape
    assert reduced_dimension <= columns
    x_mean = 1.0 / rows * np.sum(data, axis=0)
    decentralise_x = data - x_mean  # 去中心化
    cov = decentralise_x.T.dot(decentralise_x)  # 计算协方差
    eigenvalues, feature_vectors = np.linalg.eig(cov)  # 特征值分解
    min_d = np.argsort(eigenvalues)
    # 选取最大的特征值对应的特征向量
    feature_vectors = np.delete(feature_vectors, min_d[:columns - reduced_dimension], axis=1)
    return decentralise_x, feature_vectors, x_mean

'''
读入人脸数据
'''
def read_faces(file_path):
    file_list = os.listdir(file_path)
    data = []
    i = 1
    plt.figure(figsize=size)
    for file in file_list:
        path = os.path.join(file_path, file)
        plt.subplot(3, 3, i)
        with open(path) as f:
            img = cv2.imread(path) # 读取图像
            img = cv2.resize(img, size) # 压缩图像至size大小
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 三通道转换为灰度图
            plt.imshow(img_gray) # 预览
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w) # 对(h,w)的图像数据拉平
            data.append(img_col)
        i += 1
        if i>=12:
            break
    plt.show()
    return np.array(data)


def draw_pca(k_list,path='data/'):
    data = read_faces(path)
    n_samples, n_features = data.shape
    for k in k_list:
        c_data, eigVectsReduce, data_mean = pca(data, k)  # PCA降维
        print(eigVectsReduce)
        eigVectsReduce = np.real(eigVectsReduce)  # 一旦降维维度超过某个值，特征向量矩阵将出现复向量，对其保留实部
        pca_data = np.dot(c_data, eigVectsReduce)  # 计算降维后的数据
        recon_data = np.dot(pca_data, eigVectsReduce.T) + data_mean  # 重构数据
        plt.figure(figsize=size)
        for i in range(n_samples):
            plt.subplot(3, 3, i + 1)
            plt.title('k:'+str(k))
            plt.imshow(recon_data[i].reshape(size))
        plt.show()

        print("信噪比如下：")
        for i in range(n_samples):
            a = psnr(data[i], recon_data[i])
            print('图', i, '的信噪比: ', a)


""" 计算信噪比 """
def psnr(source, target):
    diff_sqrt = (source - target)**2
    rmse = np.sqrt(np.mean(diff_sqrt))
    return 20 * np.log10(255.0 / rmse)


if __name__ == '__main__':
    K_ = [5,10,20,40,80,160]
    draw_pca(K_)







