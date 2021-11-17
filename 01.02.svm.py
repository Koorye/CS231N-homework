# %%

import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets

random.seed(999)
np.random.seed(999)

# %%

train_data = datasets.MNIST(root='data', train=True)
test_data = datasets.MNIST(root='data', train=False)

# %%

X_train, y_train = [], []
indexs = np.arange(len(train_data))
np.random.shuffle(indexs)
# indexs = indexs[:1000]
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
# indexs = indexs[:100]
for i in indexs:
    data, label = test_data[i]
    X_test.append(np.array(data).flatten())
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test.shape, y_test.shape

# %%


class SVM:
    def __init__(self) -> None:
        self.history_losses = []
        return

    def fit(self, X, y, epochs=1000, lr=1e-3, batch_size=256, verbose=True):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : param y: <darray> 标签 (n_samples,) -> [cls1,cls2,...]
        : param epochs: <int> 训练次数
        : param lr: <int> 学习率
        : param verbose: <bool> 是否显示过程
        """
        self.X = X
        self.y = y

        n_samples, n_dim = self.X.shape
        n_cls = np.max(self.y) + 1
        self.W = .001 * np.random.randn(n_dim, n_cls)

        for ep in range(epochs):
            total_loss = 0.
            batch_index = np.random.choice(n_samples, batch_size)
            X_, y_ = self.X[batch_index], self.y[batch_index]
            loss, grad = self.cal_loss_(X_, y_)
            self.history_losses.append(loss)
            self.W -= lr * grad

            if verbose and ep % 50 == 0:
                print(f'第{ep}次训练，loss = {loss}')

        return self.history_losses

    def predict(self, X):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : return y_pred: <darray> 预测标签 (n_samples,) -> [cls1,cls2,...]
        """
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : param y: <darray> 标签 (n_samples,) -> [cls1,cls2,...]
        : return: <int> 准确率        
        """
        y_pred = self.predict(X)
        return (y == y_pred).sum() / len(y)

    def cal_loss_(self, X, y):
        n_samples = X.shape[0]
        # (n_samples,n_dim) @ (n_dim,n_cls)
        # -> (n_samples,n_cls)
        scores = X.dot(self.W)

        # 依次取出scores[0,y[0]],scores[1,y[1]],
        # ...,scores[n_samples,y[n_samples]]的值组成列表
        # -> (n_samples,) -> (n_samples,1)
        # (n_samples,n_cls) - (n_samples,1)
        # -> (n_samples,n_cls) - (n_samples,n_cls)
        margin = scores - scores[range(n_samples), y].reshape(-1, 1) + 1
        margin[range(n_samples), y] = 0

        # 去除小于0的元素
        margin = (margin > 0) * margin
        loss = margin.sum() / n_samples

        # 计算梯度
        # \nabla_{w_{yi}}{Li}
        # -> -[\sum_{j!=yi} 1(margin > 0)]{xi}
        # 即对于第i个样本，其特征为xi，类别为yi
        # 考虑所有其他类别的margin，记共有 m 个margin > 0
        # 则dW[:,yi] -= m * xi
        #
        # \nabla_{wj}{Li}
        # -> 1(margin > 0){xi}
        # 即对于第i个样本，其特征为xi，类别为yi
        # 对于每个类别j，如果其margin > 0
        # 则dW[:,j] += xi
        # 其中1表示示性函数，若函数内计算结果>0，返回1，否则返回0

        # 先计算\nabla_{wj}
        # 将所有margin > 0的位置设为1
        # 即对于每个margin > 0的位置(i,j)
        # 对于任意维度k，都有dW[k,j] += X[i,j]
        # (i,j,k分别表示样本、类别、维度)
        # 从而实现dW[:,j] += xi
        counts = (margin > 0).astype(int)

        # 之后计算\nabla_{w_{yi}}
        # 统计每个样本margin > 0的数量m，取-m填入对应的正确类别yi中
        # 即对于每个正确类别的位置(i,yi)
        # 对于任意维度k，都有dW[k,yi] -= m * X[i,yi]
        # 从而实现dW[:,yi] -= m * xi
        counts[range(n_samples), y] = -np.sum(counts, axis=1)

        # X.T -> (n_dim,n_samples)
        # counts -> (n_samples,n_cls)
        # dW[m,n] -> \sum_{i=1}^{n_samples} (X.T[m,i]*counts[i,n])
        # 即第m维第n类对应的dW
        # 即为第m维每个样本的特征和对应counts的元素的乘积和
        # 从而实现dW的累加和累减
        dW = X.T.dot(counts) / n_samples

        return loss, dW

# %%

svm = SVM()
losses = svm.fit(X_train, y_train, lr=1e-6, epochs=2000, batch_size=1024)
plt.plot(range(2000), losses)

# %%

svm.score(X_test, y_test)
