import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math

class Discriminator_head(nn.Module):
	"""Discriminator network with PatchGAN."""
	def __init__(self, n_class, ndf=32, use_sigmoid=False):
		super(Discriminator_head, self).__init__()
		self.n_class = n_class
		self.head = nn.Sequential(
			nn.Conv2d(ndf * 16, ndf * 16, 3, 1, 1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(ndf * 16, self.n_class, 4, bias=False)
		)
	def forward(self, x):
		output = self.head(x)
		return output.view(-1, self.n_class)

class Discriminator(nn.Module):
	"""Discriminator network with PatchGAN."""
	def __init__(self, image_c=1, ndf=32, head=[]):
		super(Discriminator, self).__init__()
		self.head = head

		self.main = nn.Sequential(
			# input is (nc) x 128 x 128
			nn.Conv2d(image_c, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 64 x 64
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 32 x 32
			nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 16 x 16
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 8 x 8
			nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*16) x 4 x 4
			# nn.Conv2d(ndf * 16, ndf*32, 4, 2, 1, bias=False),
			# nn.LeakyReLU(0.2, inplace=True),
			# (ndf*32) x 2 x 2
			)
		if len(head)==1:
			self.head0 = Discriminator_head(1, ndf)
		elif len(head)==2:
			self.head1 = Discriminator_head(head[0], ndf)
			self.head2 = Discriminator_head(head[1], ndf)
		else:
			self.head0 = Discriminator_head(1, ndf)
			self.head1 = Discriminator_head(head[1], ndf)
			self.head2 = Discriminator_head(head[2], ndf)

	def forward(self, x):
		feat = self.main(x)
		if len(self.head)==1:
			out = self.head0(feat)
			if self.head[0]: out = F.sigmoid(out)
			return [out.view(-1)]
		elif len(self.head)==2:
			out1 = self.head1(feat)
			out2 = self.head2(feat)
			return [out1.view(-1, self.head[0]), out2.view(-1, self.head[1])]
		else:
			out0 = self.head0(feat)
			if self.head[0]: out0 = F.sigmoid(out0)
			out1 = self.head1(feat)
			out2 = self.head2(feat)
			return [out0.view(-1), out1.view(-1, self.head[1]), out2.view(-1, self.head[2])]		
