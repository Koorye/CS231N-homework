# %%

import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.lib.npyio import NpzFile
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets
import tqdm
import ssl

# 取消证书验证，否则无法下载数据集
ssl._create_default_https_context = ssl._create_unverified_context
random.seed(999)
np.random.seed(999)

# %%

train_data = datasets.MNIST(root='data', train=True, download=True)
test_data = datasets.MNIST(root='data', train=False, download=True)
data, label = train_data[0]
plt.imshow(data)

# %%

from collections import Counter

class KNNClassifier:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        n_preds = X.shape[0]
        y_pred = np.zeros(n_preds).astype(int)
        for i in range(n_preds):
            dist = np.sqrt(((self.X - X[i]) ** 2).sum(axis=1))
            top_k_index = dist.argsort()[:self.k]
            most_common_label = Counter(self.y[top_k_index]).most_common(1)[0][0]
            y_pred[i] = most_common_label
        return np.array(y_pred)
    
    def score(self,X,y):
        y_pred = self.predict(X)
        return (y_pred == y).sum() / len(y)

# %%

X_train, y_train = [], []
indexs = np.arange(len(train_data))
np.random.shuffle(indexs)
indexs = indexs[:1000]
for i in indexs:
    data, label = train_data[i]
    X_train.append(np.array(data).flatten())
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train.shape, y_train.shape

# %%

X_test, y_test = [], []
indexs = np.arange(len(test_data))
np.random.shuffle(indexs)
indexs = indexs[:100]
for i in indexs:
    data, label = test_data[i]
    X_test.append(np.array(data).flatten())
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test.shape, y_test.shape

# %%

ks = [i for i in range(1,10+1)]
accs_train, accs_test = [], []
pbar = tqdm.tqdm(ks, total=len(ks))
for k in pbar:
    knn = KNNClassifier(k=k)
    # knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(X_train, y_train)
    acc_train = knn.score(X_train, y_train)
    acc_test = knn.score(X_test, y_test)
    accs_train.append(acc_train)
    accs_test.append(acc_test)

plt.plot(ks, accs_train, label='Train ACC')
plt.plot(ks, accs_test, label='Test ACC')
plt.legend()

# %%

iris = load_iris()
X, y = iris['data'], iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y)

ks = [i for i in range(1,20+1)]
accs_train, accs_test = [], []
pbar = tqdm.tqdm(ks, total=len(ks))
for k in pbar:
    knn = KNNClassifier(k=k)
    # knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    knn.fit(X_train, y_train)
    acc_train = knn.score(X_train, y_train)
    acc_test = knn.score(X_test, y_test)
    accs_train.append(acc_train)
    accs_test.append(acc_test)

plt.plot(ks, accs_train, label='Train ACC')
plt.plot(ks, accs_test, label='Test ACC')
plt.legend()

# %%
