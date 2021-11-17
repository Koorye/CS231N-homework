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

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        """
        : param input_size: 输入层的数量
        : param hidden_size: 隐藏层的数量
        : param output_size: 输出层的数量
        """
        self.params = {}
        self.params['W1'] = 1e-4 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 1e-4 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.history_losses = []
    
    def fit(self,X,y,epochs,lr=1e-3,batch_size=64,verbose=True):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : param y: <darray> 标签 (n_samples,) -> [cls1,cls2,...]
        : param epochs: <int> 训练次数
        : param lr: <int> 学习率
        : param verbose: <bool> 是否显示过程
        """
        self.X = X
        self.y = y
        n_samples = self.X.shape[0]

        for ep in range(epochs):
            batch_index = np.random.choice(n_samples, batch_size)
            X_, y_ = self.X[batch_index], self.y[batch_index]
            # loss, grads = self.cal_loss_(X_, y_)
            loss, grads = self.cal_loss_(X_, y_)
            self.history_losses.append(loss)
            self.params['W1'] -= lr * grads['W1']
            self.params['b1'] -= lr * grads['b1']
            self.params['W2'] -= lr * grads['W2']
            self.params['b2'] -= lr * grads['b2']

            if verbose and ep % 50 == 0:
                print(f'第{ep}次训练，loss = {loss}')

        return self.history_losses

    def predict(self, X):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : return y_pred: <darray> 预测标签 (n_samples,) -> [cls1,cls2,...]
        """
        hidden_in = X.dot(self.params['W1']) + self.params['b1']
        hidden_out = np.maximum(0, hidden_in)
        scores = hidden_out.dot(self.params['W2']) + self.params['b2']
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        """
        : param X: <darray> 样本 (n_samples,n_dim) -> [[dim1,dim2,...]]
        : param y: <darray> 标签 (n_samples,) -> [cls1,cls2,...]
        : return: <int> 准确率        
        """
        y_pred = self.predict(X)
        return (y == y_pred).sum() / len(y)

    def cal_loss_(self,X,y):
        """
        : param X: 输入样本 (n_samples,n_dim)
        : param y: 输出 (n_samples,)
        : return: loss, grads 损失和各参数的梯度
        """
        W1, b1 = self.params['W1'], self.params['b1']        
        W2, b2 = self.params['W2'], self.params['b2']        
        n_samples = X.shape[0]

        # 第一层正向传播 input -> hidden
        # (n_samples,n_dim) @ (n_dim,n_hidden) + (n_hidden,)
        # -> (n_samples,n_hidden)
        hidden_in = X.dot(W1) + b1
        # RELU
        # (n_samples,n_hidden)
        hidden_out = np.maximum(0, hidden_in)

        # 第二层正向传播 hidden -> output
        # (n_samples,n_hidden) @ (n_hidden,n_output) + (n_output)
        # -> (n_samples,n_output)
        scores = hidden_out.dot(W2) + b2

        # 计算Cross Entropy Loss -> Softmax + NLL Loss
        f = scores - np.max(scores, axis=1).reshape(n_samples,1)
        p = np.exp(f) / np.exp(f).sum(axis=1).reshape(n_samples,1)
        y_mask = np.zeros_like(p)
        y_mask[range(n_samples),y] = 1
        loss = -(np.log(p) * y_mask).sum() / n_samples

        grads = {}
        # 第一层反向传播
        # Softmax
        # (n_samples,n_output)
        dscores = (p-y_mask) / n_samples
 
        # (n_hidden,n_samples) @ (n_samples,n_output)
        # -> (n_hidden,n_output)
        grads['W2'] = hidden_out.T.dot(dscores)
        # (n_output,)
        grads['b2'] = dscores.sum(axis=0)

        # 第二层反向传播
        # (n_sample,n_output) @ (n_output,n_hidden)
        # -> (n_samples,n_hidden)
        dhidden_out = dscores.dot(W2.T)
        # RELU
        # (n_samples,n_hidden)
        dhidden_in = dhidden_out.copy()
        dhidden_in[hidden_out <= 0] = 0
        # np.maximum(0,hidden_out)
        grads['W1'] = X.T.dot(dhidden_in)
        # (n_hidden,)
        grads['b1'] = dhidden_in.sum(axis=0)

        return loss, grads

# %%

n_dim = X_train.shape[1]
n_cls = np.max(y_train) + 1
net = TwoLayerNet(n_dim, 256, n_cls)
EPOCHS = 10000
losses = net.fit(X_train, y_train, lr=1e-4, epochs=EPOCHS, batch_size=64)
plt.plot(range(EPOCHS), losses)

# %%

net.score(X_test, y_test)

