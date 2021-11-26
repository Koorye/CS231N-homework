# %%

import copy
import math
from imageio import imread
import uuid
import os
import h5py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import urllib

import torch
from torch import nn

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


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=.1, max_len=5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        assert embed_dim % 2 == 0

        # PE(pos,2i) = sin(pos / 10000^{2i/dim})
        # PE(pos,2i+1) = cos(pos / 10000^{2i/dim})
        # 其中pos是位置，i是维度
        pe = torch.zeros(1, max_len, embed_dim)

        position = torch.arange(0, max_len).unsqueeze(1)
        exp_term = torch.exp(torch.arange(0, embed_dim, 2)
                             * -(math.log(10000) / embed_dim))
        pe[:, :, 0::2] = torch.sin(position * exp_term)
        pe[:, :, 1::2] = torch.cos(position * exp_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        N, S, D = x.shape
        output = x + self.pe[:, :S, :]
        output = self.dropout(output)
        return output

# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=.1) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0

        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim//n_heads]))

    def forward(self, q, k, v, attn_mask=None):
        N, S, D = q.shape
        N, T, D = v.shape
        H = self.n_heads

        # (N,H,S,D')
        q = self.Wq(q).view(N, S, H, D//H).permute(0, 2, 1, 3)
        # (N,H,T,D')
        k = self.Wk(k).view(N, T, H, D//H).permute(0, 2, 1, 3)
        # (N,H,T,D')
        v = self.Wv(v).view(N, T, H, D//H).permute(0, 2, 1, 3)

        # (N,H,S,D') @ (N,H,D'T)
        # -> N,H (S,D') @ (D',T)
        # -> (N,H,S,T)
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale

        if attn_mask is not None:
            energy = energy.masked_fill(attn_mask == 0, -float('inf'))

        attention = self.dropout(torch.softmax(energy, dim=-1))

        # (N,H,S,T) @ (N,H,T,D')
        # -> N,H (S,T) @ (T,D')
        # -> (N,H,S,D')
        output = torch.matmul(attention, v).permute(
            0, 2, 1, 3).contiguous().view(N, -1, D)
        output = self.Wo(output)

        return output

# %%


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, dim_feedforward=2048, dropout=.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(input_dim, n_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, n_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None):
        # masked multi-head attn
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # multi-head attn
        tgt2 = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # feed forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

# %%


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layers) -> None:
        super().__init__()
        self.layers = self.clones(decoder_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)
        return output

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# %%



class Transformer(nn.Module):
    def __init__(self, word2idx, input_dim, wordvec_dim, n_heads=4, n_layers=2, max_len=50) -> None:
        super().__init__()

        vocab_size = len(word2idx)
        self._null = word2idx['<NULL>']
        self._start = word2idx.get('<START>', None)
        self._end = word2idx.get('<END>', None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(
            vocab_size, wordvec_dim, padding_idx=self._null)

        self.positional_encoding = PositionalEncoding(
            wordvec_dim, max_len=max_len)
        decoder_layer = TransformerDecoderLayer(wordvec_dim, n_heads)
        self.transformer = TransformerDecoder(decoder_layer, n_layers)

        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        : param features: 特征 (N,D)
        : param captions: 真实标签 (N,T)
        """
        N, T = captions.shape

        # (N,T) -> (N,T,W)
        caption_embeddings = self.embedding(captions)
        caption_embeddings = self.positional_encoding(caption_embeddings)

        # (N,D) -> (N,W) -> (N,1,W)
        projected_features = self.visual_projection(features).unsqueeze(1)

        # (T,T)
        tgt_mask = torch.tril(torch.ones(T, T,
                                         device=caption_embeddings.device,
                                         dtype=caption_embeddings.dtype))

        features = self.transformer(caption_embeddings, projected_features, tgt_mask)        
        # (N,T,W) -> (N,T,V)
        scores = self.output(features)

        return scores
    
    def sample(self, features, max_len=30):
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]

            captions = self._null * np.ones((N,max_len),dtype=np.int32)

            # (N) -> (N,1)
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).unsqueeze(1)

            for t in range(max_len):
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:,-1,:]
                # (N,V) -> (N)
                word = torch.argmax(output_logits, axis=1)

                captions[:,t] = word.numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption,word], dim=1)
            
            return captions

# %%

def temporal_softmax_loss(x,y,mask):
    N,T,V = x.shape
    x_flat = x.reshape(N*T,V)
    y_flat = y.reshape(N*T)
    mask_flat = mask.reshape(N*T)

    loss = nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = torch.mul(loss, mask_flat)
    loss = torch.mean(loss)

    return loss

# %%

data = load_coco_data(max_train=50)
word2idx = data['word_to_idx']
_null = word2idx['<NULL>']

transformer = Transformer(
    word2idx=word2idx,
    input_dim=data['train_features'].shape[1],
    wordvec_dim=256,
    n_heads=2,
    n_layers=2,
    max_len=30,
)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(.5,.999))
loss = torch.nn.CrossEntropyLoss()
history_losses = []

for ep in range(1000):
    minibatch = sample_coco_minibatch(data, batch_size=25, split='train')
    captions, features, urls = minibatch

    captions_in = captions[:, :-1]
    captions_out = captions[:,1:]

    mask = captions_out != _null

    features = torch.Tensor(features)
    captions_in = torch.LongTensor(captions_in)
    captions_out = torch.LongTensor(captions_out)
    mask = torch.LongTensor(mask)

    scores = transformer(features, captions_in)
    loss = temporal_softmax_loss(scores, captions_out, mask)
    history_losses.append(loss.item())

    optim.zero_grad()
    loss.backward()
    optim.step()

    if ep % 10 == 0:
        print(f'第{ep+1}次训练，loss = {loss.item()}')

# %%

plt.plot(history_losses)

# %%

# If you get an error, the URL just no longer exists, so don't worry!
# You can re-sample as many times as you want.
for split in ['train', 'val']:
    minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
    gt_captions, features, urls = minibatch
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])
    
    sample_captions = transformer.sample(features, max_len=30)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        img = image_from_url(url)
        # Skip missing URLs.
        if img is None: continue
        plt.imshow(img)            
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()

# %%
