# %%

import uuid
import os
import json
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from imageio import imread
from torchvision import datasets
import urllib
import tempfile

random.seed(999)
np.random.seed(999)

# %%

"""
代码来自cs231n
"""


def load_coco_data(base_dir='data/coco_captioning',
                   max_train=None,
                   pca_features=True):
    data = {}
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    f = urllib.request.urlopen(url)
    fname = str(uuid.uuid4()) + '.jpg'
    with open(fname, 'wb') as ff:
        ff.write(f.read())
    img = imread(fname)
    os.remove(fname)
    return img

# %%


data = load_coco_data(pca_features=True)

batch_size = 3
captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str)
    plt.show()

# %%


class RNN:
    def __init__(self, input_size, hidden_size) -> None:
        """
        WX: (n_input,n_hidden)
        Wh: (n_hidden,n_hidden)
        b: (n_hidden,)
        """
        self.n_input = input_size
        self.n_hidden = hidden_size

        self.WX = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.Wh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b = np.random.zeros(hidden_size,) 
        self.save_for_backwards = []

    def forward(self, X, h0):
        n_batches = X.shape[1]
        prev_h = h0
        hs = []
        for b in range(n_batches):
            next_h = self.step_forward_(X[:, b, :], prev_h)
            hs.append(next_h)
            prev_h = next_h
        hs = np.array(hs).transpose(1, 0, 2)
        return hs

    def backward(self, dh):
        n_samples, n_batches, _ = dh.shape
        dXs = []
        dWX = np.zeros((self.n_input, self.n_hidden))
        dWh = np.zeros((self.n_hidden, self.n_hidden))
        db = np.zeros(self.n_hidden,)
        for b in range(n_batches-1, -1, -1):
            dX_, d_prev_h_, dWX_, dWh_, db_ = self.step_backward_(dh[:, b, :])
            dXs.append(dX_)
            dWX += dWX_
            dWh += dWh_
            db += db_
        dh0 = d_prev_h_
        dXs = np.array(dXs[::-1]).transpose(1, 0, 2)
        return dXs, dh0, dWX, dWh, db

    def step_forward_(self, X, prev_h):
        next_h = np.tanh(np.dot(X, self.WX) + np.dot(prev_h, self.Wh) + self.b)
        self.save_for_backwards.append((X, prev_h, next_h))
        return next_h

    def step_backward_(self, dnext_h):
        X, prev_h, next_h = self.save_for_backwards.pop()
        dtanh = dnext_h * (1 - next_h**2)
        dX = np.dot(dtanh, self.WX.T)
        dprev_h = np.dot(dtanh, self.Wh.T)
        dWX = np.dot(X.T, dtanh)
        dWh = np.dot(prev_h.T, dtanh)
        db = dtanh.sum(axis=0)
        return dX, dprev_h, dWX, dWh, db

# %%


class Embedding:
    def __init__(self, input_size, output_size) -> None:
        self.n_input = input_size
        self.n_output = output_size
        self.W = 1e-2 * np.random.randn(input_size, output_size)
        self.save_for_backward = {}

    def forward(self, X):
        out = self.W[X, :]
        self.save_for_backward['X'] = X
        return out

    def backward(self, dout):
        X = self.save_for_backward['X']
        dW = np.zeros_like(self.W)
        np.add.at(dW, X, dout)
        return dW

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


class TemporalLinear:
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
        : param X: 输入样本 (n_samples,b,input_size)
        : return: out (b,output_size)
        """

        n_samples, n_batches, n_input = X.shape
        out = X.reshape(n_samples * n_batches, n_input).dot(self.W)
        out = out.reshape(n_samples, n_batches, self.output_size) + self.b
        self.save_for_backward['X'] = X
        return out

    def backward(self, dout):
        """
        : param dout: 下游梯度 (b,output_size)
        : return: dX,dW,db X,W,b的梯度
        """
        X = self.save_for_backward['X']
        n_samples, n_batches, n_input = X.shape
        dX = dout.reshape(n_samples*n_batches, self.output_size).dot(
            self.W.T).reshape(n_samples, n_batches, n_input)
        dW = dout.reshape(n_samples*n_batches, self.output_size).T.dot(
            X.reshape(n_samples*n_batches, n_input)).T
        db = np.sum(dout, axis=(0, 1))

        return dX, dW, db


linear = Linear(64, 16)
X = np.random.randn(4, 16, 64)
out = linear.forward(X)
out.shape

# %%


def softmax_loss(X, y, mask):
    """
    X: (n_samples,n_batches,n_words)
    y: (n_samples,n_batches) 0<=y<n_words
    mask: (n_samples,n_batches) 表示X[s,b]是否参与损失计算
    """
    n_samples, n_batches, n_words = X.shape
    X = X.reshape(n_samples * n_batches, n_words)
    y = y.reshape(n_samples * n_batches)
    mask = mask.reshape(n_samples * n_batches)

    p = np.exp(X-np.max(X, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)
    loss = -np.sum(mask * np.log(p[range(n_samples*n_batches), y])) / n_samples

    dX = p.copy()
    dX[range(n_samples*n_batches), y] -= 1
    dX /= n_samples
    dX *= mask[:, None]
    dX = dX.reshape(n_samples, n_batches, n_words)

    return loss, dX

# %%


word2idx = data['word_to_idx']
null = word2idx['<NULL>']
start = word2idx.get('<START>', None)
end = word2idx.get('<END>', None)

small_data = load_coco_data(max_train=50)
lr = 1e-2
loss_history = []

flatten = Flatten()
linear = Linear(512, 512)
embedding = Embedding(len(word2idx), 256)
rnn = RNN(256, 512)
tempLinear = TemporalLinear(512, len(word2idx))

for ep in range(1000):
    captions, features, urls = sample_coco_minibatch(
        small_data, batch_size=256)
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    mask = (captions_out != null)

    h0 = linear.forward(features)
    cap_in_emb = embedding.forward(captions_in)
    h = rnn.forward(cap_in_emb, h0)
    out = tempLinear.forward(h)
    loss, dout = softmax_loss(out, captions_out, mask)
    loss_history.append(loss)

    if ep % 1 == 0:
        print('ep =', ep, ', loss =', loss, ', lr =',lr)

    dX, dW, db = tempLinear.backward(dout)
    tempLinear.W -= lr * dW
    tempLinear.b -= lr * db

    dXs, dh0, dWX, dWh, db = rnn.backward(dX)
    rnn.WX -= lr * dWX
    rnn.Wh -= lr * dWh
    rnn.b -= lr * db

    dW = embedding.backward(dXs)
    embedding.W -= lr * dW

    lr *= .99

# %%

plt.plot(loss_history)
plt.show()

# %%

for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(small_data, split=split, batch_size=1)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    N = features.shape[0]
    captions = null * np.ones((N, 30), dtype=np.int32)
    prev_h = linear.forward(features)
    x = np.array([start for i in range(N)])
    captions[:, 0] = start

    for i in range(1, 30):
        x_emb = embedding.forward(x)
        next_h = rnn.step_forward_(x_emb, prev_h)
        prev_h = next_h
        out = next_h.dot(tempLinear.W) + tempLinear.b
        x = out.argmax(1)
        captions[:,i] = x

    sample_captions = decode_captions(captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
