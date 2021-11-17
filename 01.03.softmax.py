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


class SoftmaxClassifier:
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
        n_cls = self.W.shape[1]

        # (n_samples,n_dim) @ (n_dim,n_cls)
        # -> (n_samples,n_cls)
        f = X.dot(self.W)
        # 进行指数修正，减小计算量，不影响结果
        f -= f.max(axis=1).reshape(n_samples,1)
        # 计算概率
        # p_{ij} = exp(f_{ij}) / (\sum{f_{i·})
        # (i,j分别表示样本和类别)
        # (n_samples,n_cls)
        p = np.exp(f) / np.exp(f).sum(axis=1).reshape(n_samples,1)

        # 每个样本对应的类别置1
        y_mask = np.zeros((n_samples, n_cls))
        y_mask[range(n_samples),y] = 1
        # 每个样本对应类别的对数取负即为loss
        loss = -(np.log(p) * y_mask).sum() / n_samples

        # 计算梯度
        # (n_samples,n_dim) -> (n_dim,n_samples)
        # (n_dim,n_samples) @ (n_samples,n_cls)
        # -> (n_dim,n_cls)
        #
        # 记p[k,j] - 1(yi = j)为c[k,j]
        # dW_{i,j} -> dW[i,j] 
        # -> \sum_{k=1}^{n_samples} {X[i,k] * c[k,j])}
        # -> X_{i·} * C_{·j}
        # (i,j,k分别表示维度、类别、样本)
        # 即第i个维度第j个类别W的梯度为
        # 对于每个样本，其第i个维度和在第j个类别的概率(减去掩码)乘积和
        dW = X.T.dot(p-y_mask) / n_samples

        return loss, dW

# %%

sc = SoftmaxClassifier()
EPOCHS = 5000
losses = sc.fit(X_train, y_train, lr=1e-6, epochs=EPOCHS, batch_size=1024)
plt.plot(range(EPOCHS), losses)

# %%

sc.score(X_test, y_test)

# %%
