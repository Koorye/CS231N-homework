# %%

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets

random.seed(999)
np.random.seed(999)

# %%


class Conv2d:
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding=0) -> None:
        """
        : param input_size: 输入通道数
        : param output_size: 输出通道数
        : param kernel_size: 卷积核尺寸
        : param stride: 步长
        : param padding: 扩展
        """
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.save_for_backward = {}

        # (output_size,input_size,kernel_size,kernel_size)
        # 即output_size个卷积核
        # 每个卷积核尺寸为(input_size,kernel_size,kernel_size)
        bound = np.sqrt(2/(output_size * kernel_size * kernel_size))
        self.W = np.random.randn(
            output_size, input_size, kernel_size, kernel_size) * bound
        # (output_size,)
        self.b = np.random.randn(output_size) * bound

    def forward(self, X):
        """
        : param X: 输入样本 (b,c,h,w)
        : return: 输出样本 (b,output_size,h,w)
        """
        # b,c维度不填充，h,w维度填充padding个0
        X = np.pad(X, ((0,), (0,), (self.padding,),
                   (self.padding,)), mode='constant')
        b, _, h, w = X.shape

        h_out = 1 + (h-self.kernel_size) // self.stride
        w_out = 1 + (w-self.kernel_size) // self.stride
        out = np.zeros((b, self.output_size, h_out, w_out))

        # 对于输出样本的每个坐标 (i,j)
        for i in range(h_out):
            for j in range(w_out):
                # 取卷积核左上角的坐标
                # 若步长为1 -> 0,1,2,...,h_out-1
                # 若步长为2 -> 0,2,4,...,2*(h_out-1)
                # ...
                h_current = i * self.stride
                w_current = j * self.stride
                # 取所有批次、所有通道
                # 框出卷积核元素
                X_mask = X[:, :,
                           h_current: h_current+self.kernel_size,
                           w_current: w_current+self.kernel_size]
                # 遍历每个卷积核
                for k in range(self.output_size):
                    # 求卷积核框出的元素和卷积核的权重乘积和
                    out[:, k, i, j] = np.sum(X_mask * self.W[k, :, :, :])
        # (b,output_size,h,w) + (1,output_size,1,1)
        # (b,output_size,h,w) + (b,output_size,h,w)
        out = out + self.b.reshape(1, self.output_size, 1, 1)

        self.save_for_backward['X'] = X

        return out

    def backward(self, dout):
        """
        : param dout: 下游梯度 (b,output_size,h,w)
        : return: dX,dW,db X,W,b的梯度
        """

        # X已经填充过
        X = self.save_for_backward['X']
        b, _, h, w = X.shape
        h_out = 1 + (h-self.kernel_size) // self.stride
        w_out = 1 + (w-self.kernel_size) // self.stride

        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        # 将每个卷积核输出的所有批次和宽高求和
        db = np.sum(dout, axis=(0, 2, 3))

        # 对于输出样本的每个坐标 (i,j)
        for i in range(h_out):
            for j in range(w_out):
                # 取卷积核左上角的坐标
                # 若步长为1 -> 0,1,2,...,h_out-1
                # 若步长为2 -> 0,2,4,...,2*(h_out-1)
                # ...
                h_current = i * self.stride
                w_current = j * self.stride
                # 取所有批次、所有通道
                # 框出卷积核元素
                X_mask = X[:, :,
                           h_current: h_current+self.kernel_size,
                           w_current: w_current+self.kernel_size]
                # 遍历每个卷积核
                for k in range(self.output_size):
                    dW[k, :, :, :] += np.sum(X_mask * dout[:, k, i, j]
                                             [:, None, None, None], axis=0)
                for n in range(b):  # compute dx_pad
                    dX[n, :,
                       h_current:h_current+self.kernel_size,
                       w_current:w_current+self.kernel_size] += np.sum((self.W[:, :, :, :] *
                                                                        (dout[n, :, i, j])[:, None, None, None]), axis=0)
        if self.padding != 0:
            dX = dX[:, :,
                    self.padding:-self.padding,
                    self.padding:-self.padding]

        return dX, dW, db


conv = Conv2d(6, 16, 3)
X = np.random.randn(2, 6, 24, 24)
out = conv.forward(X)
conv.backward(out)[0].shape

# %%


class SpaticalConv2d:
    def __init__(self, input_c, output_c, kernel_size, stride=1, padding=0) -> None:
        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.save_for_backward = {}

        bound = np.sqrt(2/(output_c * kernel_size * kernel_size))
        self.W = np.random.randn(
            output_c, input_c, kernel_size, kernel_size) * bound
        # (output_size,)
        self.b = np.random.randn(output_c) * bound

    def forward(self, X):
        self.save_for_backward['X'] = X

        X = np.pad(X, ((0,), (0,), (self.padding,), (self.padding,)),
                   mode='constant')
        b, _, h, w = X.shape
        h_out = 1 + (h-self.kernel_size) // self.stride
        w_out = 1 + (w-self.kernel_size) // self.stride

        XC = self.im2col_(X)
        WC = self.W.reshape(-1, self.output_c)
        YC = XC.dot(WC) + self.b

        self.save_for_backward['XC'] = XC

        return YC.reshape(b, self.output_c, h_out, w_out)

    def backward(self, dout):
        X = self.save_for_backward['X']
        XC = self.save_for_backward['XC']

        db = np.sum(dout, axis=(0, 2, 3))
        dYC = dout.reshape(-1, self.output_c)
        dWC = XC.T.dot(dYC)
        dW = dWC.reshape(self.W.shape)

        dY = np.pad(dout,  (
            (0, 0),
            (0, 0),
            (self.kernel_size - 1, self.kernel_size - 1),
            (self.kernel_size - 1, self.kernel_size - 1)),
             mode='constant')
        dYC = self.im2col_(dY)

        W_flip = self.W[::-1, ...]
        W_flip = W_flip.swapaxes(2, 3)
        WC_flip = W_flip.reshape(-1, self.input_c)

        dXC = dYC.dot(WC_flip)
        dX = dXC.reshape(X.shape)

        return dX, dW, db

    def im2col_(self, X):
        XC = []
        for b in range(X.shape[0]):
            for i in range(0, X.shape[2] - self.kernel_size + 1, self.stride):
                for j in range(0, X.shape[3] - self.kernel_size + 1, self.stride):
                    col = X[b, :, i:i + self.kernel_size,
                            j:j + self.kernel_size].reshape(-1)
                    XC.append(col)
        return np.array(XC)


conv = SpaticalConv2d(1, 2, 3)
X = np.random.randn(3, 1, 5, 5)
out = conv.forward(X)
out.shape

# %%


class BatchNorm2d:
    def __init__(self, input_size, eps=1e-5, momentum=.9) -> None:
        """
        : param input_size: 输入样本的通道数，和输出样本相同
        : param eps: 用于标准化的稳定，使得方差不为0 (X-mean) / (\sqrt{var+eps})
        : param momentum: 动量，用于mean和var的更新
        """
        self.input_size = input_size
        self.eps = eps
        self.momentum = momentum

        self.mode = 'train'
        self.save_for_backward = {}
        self.gamma = np.ones((input_size,))
        self.beta = np.zeros((input_size,))
        self.running_mean = None
        self.running_var = None

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def forward(self, X):
        """
        : param X: 输入样本 (b,c,h,w)
        : return: out 输出样本 (b,c,h,w)
        """
        b, c, h, w = X.shape
        # 对每个通道进行标准化
        # (b,c,h,w) -> (b,h,w,c) -> (b*h*w,c)
        X = X.transpose(0, 2, 3, 1).reshape(b*h*w, c)
        if self.mode == 'train':
            sample_mean = np.mean(X, axis=0)
            sample_var = np.var(X, axis=0)
            X_hat = (X-sample_mean) / np.sqrt(sample_var + self.eps)
            out = self.gamma * X_hat + self.beta

            # 更新mean和var
            if self.running_mean is not None:
                self.running_mean = self.momentum * \
                    self.running_mean + (1-self.momentum) * sample_mean
                self.running_var = self.momentum * \
                    self.running_var + (1-self.momentum) * sample_var
            else:
                self.running_mean = sample_mean
                self.running_var = sample_var

            self.save_for_backward['X'] = X
            self.save_for_backward['X_hat'] = X_hat
            self.save_for_backward['sample_mean'] = sample_mean
            self.save_for_backward['sample_var'] = sample_var

        elif self.mode == 'test':
            X_hat = (X-self.running_mean) / np.sqrt(self.running_var+self.eps)
            out = self.gamma * X_hat + self.beta

        return out.reshape(b, h, w, c).transpose(0, 3, 1, 2)

    def backward(self, dout):
        """
        : param dout: 下游梯度 (b,c,h,w)
        : return: dX,dgamma,dbeta X,gamma,beta的梯度
        """
        b, c, h, w = dout.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(b*h*w, c)
        m = dout.shape[0]

        X = self.save_for_backward['X']
        X_hat = self.save_for_backward['X_hat']
        sample_mean = self.save_for_backward['sample_mean']
        sample_var = self.save_for_backward['sample_var']

        dgamma = np.sum(dout * X_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        dX_hat = dout * self.gamma
        dvar = (dX_hat * (X-sample_mean) * (-.5) *
                ((sample_var+self.eps)**-1.5)).sum(axis=0)
        dmean = (dX_hat * (-1) * ((sample_var+self.eps)**-.5)).sum(axis=0)
        dX = dX_hat * ((sample_var+self.eps)**-.5) + dvar * \
            2*(X-sample_mean) / m + dmean / m

        dX = dX.reshape(b, h, w, c).transpose(0, 3, 1, 2)

        return dX, dgamma, dbeta


bn = BatchNorm2d(16)
X = np.random.randn(2, 16, 5, 5)
out = bn.forward(X)
out.shape

# %%


class Dropout:
    def __init__(self, p=.5) -> None:
        self.p = p
        self.mode = 'train'
        self.save_for_backward = {}

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    def forward(self, X):
        """
        : param X: 输入样本
        : return: out 输出样本，与输入样本维度相同
        """
        if self.mode == 'train':
            mask = np.random.rand(*X.shape) < (1-self.p)
            mask = mask / (1-self.p)
            out = mask * X
            self.save_for_backward['mask'] = mask
        else:
            out = X

        return out

    def backward(self, dout):
        """
        : param dout: 下游梯度
        : return: dX, X的梯度
        """
        mask = self.save_for_backward['mask']
        dX = dout * mask
        return dX


dropout = Dropout()
X = np.random.randn(2, 1, 5, 5)
out = dropout.forward(X)
out.shape

# %%


class ReLU:
    def __init__(self) -> None:
        self.save_for_backward = {}

    def forward(self, X):
        out = np.maximum(0, X)
        self.save_for_backward['X'] = X
        return out

    def backward(self, dout):
        X = self.save_for_backward['X']
        dX = dout * (X > 0)
        return dX


relu = ReLU()
X = np.random.randn(2, 1, 5, 5)
out = relu.forward(X)
relu.backward(out)
# out

# %%


class Flatten:
    def __init__(self) -> None:
        self.save_for_backward = {}

    def forward(self, X):
        self.save_for_backward['shape'] = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        shape = self.save_for_backward['shape']
        dX = dout.reshape(*shape)
        return dX


flatten = Flatten()
X = np.random.randn(3, 1, 5, 5)
out = flatten.forward(X)
out.shape

# %%


class Linear:
    def __init__(self, input_size, output_size) -> None:
        """
        : param input_size: 输入维度
        : param output_size: 输出维度
        """
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(
            input_size, output_size) * np.sqrt(2/input_size)
        self.b = np.zeros(output_size)
        self.save_for_backward = {}

    def forward(self, X):
        """
        : param X: 输入样本 (b,input_size)
        : return: out (b,output_size)
        """
        out = X.dot(self.W) + self.b
        self.save_for_backward['X'] = X
        return out

    def backward(self, dout):
        """
        : param dout: 下游梯度 (b,output_size)
        : return: dX,dW,db X,W,b的梯度
        """
        X = self.save_for_backward['X']
        dX = dout.dot(self.W.T)
        dW = X.T.dot(dout)
        db = np.sum(dout, axis=0)

        return dX, dW, db


linear = Linear(64, 16)
X = np.random.randn(2, 64)
out = linear.forward(X)
out.shape

# %%


class CNN:
    def __init__(self) -> None:
        self.conv1 = SpaticalConv2d(1, 6, 5)
        self.bn1 = BatchNorm2d(6)
        self.relu1 = ReLU()

        self.conv2 = SpaticalConv2d(6, 16, 3)
        self.bn2 = BatchNorm2d(16)
        self.relu2 = ReLU()

        self.flatten = Flatten()

        self.fc1 = Linear(7744, 256)
        # self.fc1 = Linear(784, 256)
        self.relu3 = ReLU()

        self.fc2 = Linear(256, 10)

    def forward(self, X):
        X = self.conv1.forward(X)
        X = self.bn1.forward(X)
        X = self.relu1.forward(X)

        X = self.conv2.forward(X)
        X = self.bn2.forward(X)
        X = self.relu2.forward(X)

        X = self.flatten.forward(X)

        X = self.fc1.forward(X)
        X = self.relu3.forward(X)

        X = self.fc2.forward(X)

        return X

    def backward(self, dout):
        grads = {}

        dX, dW, db = self.fc2.backward(dout)
        grads['fc2'] = {}
        grads['fc2']['dW'] = dW
        grads['fc2']['db'] = db

        # dX = self.relu3.backward(dX)
        dX, dW, db = self.fc1.backward(dX)
        grads['fc1'] = {}
        grads['fc1']['dW'] = dW
        grads['fc1']['db'] = db

        dX = self.flatten.backward(dX)

        dX = self.relu2.backward(dX)
        dX, dgamma, dbeta = self.bn2.backward(dX)
        grads['bn2'] = {}
        grads['bn2']['dgamma'] = dgamma
        grads['bn2']['dbeta'] = dbeta

        dX, dW, db = self.conv2.backward(dX)
        grads['conv2'] = {}
        grads['conv2']['dW'] = dW
        grads['conv2']['db'] = db

        dX = self.relu1.backward(dX)
        dX, dgamma, dbeta = self.bn1.backward(dX)
        grads['bn1'] = {}
        grads['bn1']['dgamma'] = dgamma
        grads['bn1']['dbeta'] = dbeta

        _, dW, db = self.conv1.backward(dX)
        grads['conv1'] = {}
        grads['conv1']['dW'] = dW
        grads['conv1']['db'] = db

        return grads


cnn = CNN()
X = np.random.randn(64, 1, 28, 28)
out = cnn.forward(X)
cnn.backward(out)

# %%

train_data = datasets.MNIST(root='data', train=True)
test_data = datasets.MNIST(root='data', train=False)

# %%

X_train, y_train = [], []
indexs = np.arange(len(train_data))
np.random.shuffle(indexs)
indexs = indexs[:1000]
for i in indexs:
    data, label = train_data[i]
    X_train.append(np.array(data)[None, :, :]/255.)
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
    X_test.append(np.array(data)[None, :, :]/255.)
    y_test.append(label)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test.shape, y_test.shape

# %%


def normalize(X, mean=.5, std=.5):
    return (X-mean)/std


X_train = normalize(X_train)
X_test = normalize(X_test)

# %%

cnn = CNN()
EPOCHS = 5000
BATCH_SIZE = 1
LR = 1e-4
n_samples = X_train.shape[0]
history_losses = []

total_loss = 0.
for ep in range(EPOCHS):
    batch_index = np.random.choice(n_samples, BATCH_SIZE)
    X, y = X_train[batch_index], y_train[batch_index]
    scores = cnn.forward(X)

    # 计算Cross Entropy Loss -> Softmax + NLL Loss
    f = scores - np.max(scores, axis=1).reshape(BATCH_SIZE, 1)
    p = np.exp(f) / np.exp(f).sum(axis=1).reshape(BATCH_SIZE, 1)
    y_mask = np.zeros_like(p)
    y_mask[range(BATCH_SIZE), y] = 1
    loss = -(np.log(p) * y_mask).sum() / BATCH_SIZE
    total_loss += loss
    history_losses.append(total_loss / (ep+1))

    dscores = (p-y_mask) / BATCH_SIZE
    grads = cnn.backward(dscores)

    cnn.conv1.W -= LR * grads['conv1']['dW']
    cnn.conv1.b -= LR * grads['conv1']['db']
    cnn.bn1.gamma -= LR * grads['bn1']['dgamma']
    cnn.bn1.beta -= LR * grads['bn1']['dbeta']
    cnn.conv2.W -= LR * grads['conv2']['dW']
    cnn.conv2.b -= LR * grads['conv2']['db']
    cnn.bn2.gamma -= LR * grads['bn2']['dgamma']
    cnn.bn2.beta -= LR * grads['bn2']['dbeta']
    cnn.fc1.W -= LR * grads['fc1']['dW']
    cnn.fc1.b -= LR * grads['fc1']['db']
    cnn.fc2.W -= LR * grads['fc2']['dW']
    cnn.fc2.b -= LR * grads['fc2']['db']

    if ep % 10 == 0:
        print(f'第{ep}次训练，avg loss = {total_loss/(ep+1)}')

# %%

plt.plot(history_losses)

# %%


cnn = torch.nn.Sequential(
    torch.nn.Conv2d(1, 6, 5),
    torch.nn.BatchNorm2d(6),
    torch.nn.ReLU(),

    torch.nn.Conv2d(6, 16, 3),
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),

    torch.nn.Flatten(),

    torch.nn.Linear(7744, 256),
    torch.nn.ReLU(),

    torch.nn.Linear(256, 10),
)

EPOCHS = 1000
BATCH_SIZE = 1
LR = 1e-4
n_samples = X_train.shape[0]
history_losses = []

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=cnn.parameters(), lr=1e-4, momentum=0)

total_loss = 0.
for ep in range(EPOCHS):
    batch_index = np.random.choice(n_samples, BATCH_SIZE)
    X, y = X_train[batch_index], y_train[batch_index]
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    scores = cnn(X)

    loss = criterion(scores, y)
    total_loss += loss.item()
    history_losses.append(total_loss / (ep+1))

    optim.zero_grad()
    loss.backward()
    optim.step()

    if ep % 1 == 0:
        print(f'第{ep}次训练，avg loss = {total_loss/(ep+1)}')

# %%

plt.plot(history_losses)
