# %%

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # Set default size of plots.
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images):
    # Images reshape to (batch_size, D).
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return

# %%


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

# %%


NUM_TRAIN = 50000
NUM_VAL = 5000

NOISE_DIM = 96
batch_size = 128

mnist_train = dset.MNIST(
    './cs231n/datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_train = DataLoader(
    mnist_train,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_TRAIN, 0)
)

mnist_val = dset.MNIST(
    './cs231n/datasets/MNIST_data',
    train=True,
    download=True,
    transform=T.ToTensor()
)
loader_val = DataLoader(
    mnist_val,
    batch_size=batch_size,
    sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
)

imgs = loader_train.__iter__().next()[0].view(
    batch_size, 784).numpy().squeeze()
show_images(imgs)

# %%


def sample_noise(batch_size, dim, seed=None):
    """
    Generate a PyTorch Tensor of uniform random noise.
    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    if seed is not None:
        torch.manual_seed(seed)
    return 2 * torch.rand(batch_size, dim) - 1


sample_noise(64, 100).size()

# %%


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        # "flatten" the C * H * W values into a single vector per image
        return x.view(N, -1)


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)

# %%


NOISE_DIM = 96


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# %%


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    N = logits_real.size()
    real_labels = torch.ones(N)
    fake_labels = 1 - real_labels
    loss = bce_loss(logits_real, real_labels) + \
        bce_loss(logits_fake, fake_labels)

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.
    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    N = logits_fake.size()
    fake_labels = torch.ones(N)
    loss = bce_loss(logits_fake, fake_labels)

    return loss

# %%


noise_size = 96

D = Discriminator()
G = Generator()

D_optim = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
G_optim = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

images = []

for ep in range(10):
    for data, _ in loader_train:
        if len(data) != batch_size:
            continue

        D_optim.zero_grad()
        logits_real = D(2 * (data - 0.5))

        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed).detach()
        logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

        d_total_error = discriminator_loss(logits_real, logits_fake)
        d_total_error.backward()
        D_optim.step()

        G_optim.zero_grad()
        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed)

        gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
        g_error = generator_loss(gen_logits_fake)
        g_error.backward()
        G_optim.step()

    if ep % 1 == 0:
        print(f'第{ep+1}次训练，D loss = {d_total_error.item()}，G loss = {g_error.item()}')
        imgs_numpy = fake_images.data.cpu().numpy()
        images.append(imgs_numpy[0:16])

# %%

numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 1
    print()

# %%


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss_real = (0.5 * (scores_real - 1) ** 2).mean()
    loss_fake = (0.5 * scores_fake ** 2).mean()
    loss = loss_real + loss_fake
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = (0.5 * (scores_fake - 1) ** 2).mean()
    return loss

# %%


D_optim = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
G_optim = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

images = []

for ep in range(10):
    for data, _ in loader_train:
        if len(data) != batch_size:
            continue

        D_optim.zero_grad()
        logits_real = D(2 * (data - 0.5))

        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed).detach()
        logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

        d_total_error = ls_discriminator_loss(logits_real, logits_fake)
        d_total_error.backward()
        D_optim.step()

        G_optim.zero_grad()
        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed)

        gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
        g_error = ls_generator_loss(gen_logits_fake)
        g_error.backward()
        G_optim.step()

    if ep % 1 == 0:
        print(f'第{ep+1}次训练，D loss = {d_total_error.item()}，G loss = {g_error.item()}')
        imgs_numpy = fake_images.data.cpu().numpy()
        images.append(imgs_numpy[0:16])

# %%

numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 1
    print()

# %%


class DcDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            Unflatten(batch_size, 1, 28, 28),
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(64*4*4, 4*4*64),
            nn.LeakyReLU(),
            nn.Linear(4*4*64, 1)
        )

    def forward(self, x):
        return self.model(x)


class DcGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(),
            nn.BatchNorm1d(7*7*128),
            Unflatten(-1, 128, 7, 7),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh(),
            Flatten(),
        )

    def forward(self, x):
        return self.model(x)

# %%


noise_size = 96

D = DcDiscriminator()
G = DcGenerator()

D.apply(initialize_weights)
G.apply(initialize_weights)

D_optim = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
G_optim = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

images = []

for ep in range(5):
    for data, _ in loader_train:
        if len(data) != batch_size:
            continue

        D_optim.zero_grad()
        logits_real = D(2 * (data - 0.5))

        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed).detach()
        logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

        d_total_error = discriminator_loss(logits_real, logits_fake)
        d_total_error.backward()
        D_optim.step()

        G_optim.zero_grad()
        g_fake_seed = sample_noise(batch_size, noise_size)
        fake_images = G(g_fake_seed)

        gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
        g_error = generator_loss(gen_logits_fake)
        g_error.backward()
        G_optim.step()

    if ep % 1 == 0:
        print(f'第{ep+1}次训练，D loss = {d_total_error.item()}，G loss = {g_error.item()}')
        imgs_numpy = fake_images.data.cpu().numpy()
        images.append(imgs_numpy[0:16])

# %%

numIter = 0
for img in images:
    print("Iter: {}".format(numIter))
    show_images(img)
    plt.show()
    numIter += 1
    print()

# %%