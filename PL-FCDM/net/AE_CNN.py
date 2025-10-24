import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, p1, p2, c1, c2, h1, e1, dim=116, activation=nn.Tanh(), channel=1, re_activation=False, instance_norm2=False):
        super(Autoencoder, self).__init__()

        self.re_activation = re_activation
        self.instance_norm2 = instance_norm2
        self.p1 = p1
        self.p2 = p2

        # Encoder
        self.encoder_conv1 = nn.Conv2d(channel, c1, (1, dim), stride=1, padding=0)
        self.encoder_conv2 = nn.Conv2d(c1, c2, (dim, 1), stride=1, padding=0)
        self.encoder_activation = activation
        self.encoder_norm1 = nn.InstanceNorm2d(c1, affine=False)
        self.encoder_norm2 = nn.InstanceNorm2d(c2, affine=False)
        self.encoder_flatten = nn.Flatten()
        self.encoder_linear1 = nn.Linear(c2, h1)
        self.encoder_linear2 = nn.Linear(h1, e1)

        # Decoder
        self.decoder_linear1 = nn.Linear(e1, h1)
        self.decoder_linear2 = nn.Linear(h1, c2)
        self.decoder_unflatten = nn.Unflatten(1, (c2, 1, dim))
        self.decoder_conv1 = nn.Conv2d(c2, c1, (dim, 1), stride=1, padding=0)
        self.decoder_conv2 = nn.Conv2d(c1, channel, (1, dim), stride=1, padding=0)
        self.decoder_activation = activation
        self.decoder_norm1 = nn.InstanceNorm2d(c1, affine=False)
        self.decoder_norm2 = nn.InstanceNorm2d(channel, affine=False)

        self.dropout1 = nn.Dropout(p1)
        self.dropout2 = nn.Dropout(p2)
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        x = x.to(torch.float32)
        x = self.encoder_conv1(x)
        x = self.encoder_activation(x)
        x = self.encoder_norm1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_activation(x)
        if self.instance_norm2:
            x = self.encoder_norm2(x)
        x = self.encoder_flatten(x)
        x = self.encoder_linear1(x)
        x = self.encoder_activation(x)
        x = self.dropout1(x)
        x = self.encoder_linear2(x)
        return x

    def decode(self, z):
        z = self.decoder_linear1(z)
        z = self.decoder_activation(z)
        z = self.dropout2(z)
        z = self.decoder_linear2(z)
        z = self.decoder_unflatten(z)
        z = self.decoder_conv1(z)
        z = self.decoder_activation(z)
        z = self.decoder_norm1(z)
        z = self.decoder_conv2(z)
        z = self.decoder_activation(z)
        z = self.decoder_norm2(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed