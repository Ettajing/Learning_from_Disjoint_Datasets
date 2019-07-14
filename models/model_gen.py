import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math

image_size = 128
class ResidualBlock(nn.Module):
	''' residual block with spatchnormalization'''
	def __init__(self, dim_in, dim_out, norm):
		super(ResidualBlock, self).__init__()
		norm_layer = nn.InstanceNorm2d
		if norm=='batch':
			norm_layer = nn.BatchNorm2d
		self.main = nn.Sequential(
			nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
			norm_layer(dim_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
			norm_layer(dim_out))
	def forward(self, x):
		return x + self.main(x)

class IBNBlock(nn.Module):
	def __init__(self, dim):
		super(IBNBlock, self).__init__()
		self.dim_half = dim//2
		self.batchnorm = nn.BatchNorm2d(self.dim_half)
		self.instancenorm = nn.InstanceNorm2d(self.dim_half)
	def forward(self, x):
		x1 = self.batchnorm(x.narrow(1, 0, self.dim_half).contiguous())
		x2 = self.instancenorm(x.narrow(1,self.dim_half, self.dim_half))
		return torch.cat([x1, x2], 1)

class Encoder(nn.Module):
	def __init__(self, img_c=1, conv_dim=32, dim_attr=64, dim_id=128, dim_source=64, norm='instance'):
		super(Encoder, self).__init__()
		if norm =='batch':
			norm_layer = nn.BatchNorm2d
		elif norm=='instance':
			norm_layer =  nn.InstanceNorm2d
		else:
			norm_layer = IBNBlock
		self.dim_source = dim_source

		#-----------------------enc-------------------------
		layers = [nn.Conv2d(img_c, conv_dim, kernel_size=5, stride=1, padding=2),
			norm_layer(conv_dim), nn.ReLU(inplace=True)]

		# Down-sampling layers.
		curr_dim = conv_dim
		for i in range(4):
			layers.append(nn.Conv2d(curr_dim, curr_dim+conv_dim, kernel_size=4, stride=2, padding=1))
			layers.append(norm_layer(curr_dim+conv_dim))
			layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim + conv_dim

		# Bottleneck layers.
		self.enc_conv = nn.Sequential(*layers)
		curr_dim *= (image_size//2**4)**2
		self.enc_fc1 = nn.Linear(curr_dim, dim_attr)
		self.enc_fc2 = nn.Linear(curr_dim, dim_attr)
		self.enc_fc3 = nn.Linear(curr_dim, dim_id)
		self.enc_fc4 = nn.Linear(curr_dim, dim_source)

	def forward(self, x):
		# Replicate spatially and concatenate domain information.
		x = self.enc_conv(x)
		x = x.view(x.size(0), -1)
	
		illu_feat = F.normalize(self.enc_fc1(x))
		pose_feat = F.normalize(self.enc_fc2(x))
		id_feat = F.normalize(self.enc_fc3(x))
		source_feat = F.normalize(self.enc_fc4(x)) 
	
		return illu_feat, pose_feat, id_feat, source_feat
		
class Decoder(nn.Module):
	def __init__(self, img_c=1, conv_dim=32, dim_attr=64, dim_id=128, dim_source=128, norm='instance'):
		super(Decoder, self).__init__()
		if norm =='batch':
			norm_layer = nn.BatchNorm2d
		elif norm=='instance':
			norm_layer =  nn.InstanceNorm2d
		else:
			norm_layer = IBNBlock
		self.dim_source = dim_source
		layers = []
		curr_dim = 5*conv_dim
		for i in range(4):
			layers.append(nn.ConvTranspose2d(curr_dim, curr_dim-conv_dim, kernel_size=4, stride=2, padding=1))
			layers.append(norm_layer(curr_dim-conv_dim))
			layers.append(nn.ReLU(inplace=True))
			curr_dim = curr_dim - conv_dim
		layers.append(nn.Conv2d(curr_dim, img_c, kernel_size=3, stride=1, padding=1))
		layers.append(nn.Tanh())
		
		self.dec_conv = nn.Sequential(*layers)
		self.dec_fc = nn.Linear(dim_attr*2 + dim_id+dim_source, 5*conv_dim*8*8)
		self.med_dim = conv_dim*5
		
	def forward(self, x1, x2, x3, x4):
		x = torch.cat([x1, x2, x3, x4], dim=1) if self.dim_source>0 else torch.cat([x1, x2, x3], dim=1)
		x = self.dec_fc(x)
		x = x.view(x.size(0), self.med_dim, 8, 8)
		return self.dec_conv(x)

class Generator(nn.Module):
	"""Generator network."""
	def __init__(self, img_c=1, conv_dim=32, dim_attr=64, dim_id=128, dim_source=128, norm='batch'):
		super(Generator, self).__init__()
		self.enc = Encoder(img_c, conv_dim, dim_attr, dim_id, dim_source, norm)
		self.dec = Decoder(img_c, conv_dim, dim_attr, dim_id, dim_source, norm)

	def forward_enc(self, x):
		return self.enc(x) 

	def _init_weights(self):
		''' Set weights to Gaussian, biases to zero '''
		print('init weights')
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				#m.bias.data.zero_() + 1
				m.bias.data = torch.ones(m.bias.data.size())

	def forward_dec(self, x1, x2, x3, x4):
		return self.dec(x1, x2, x3, x4)

	def forward(self, x):
		f1, f2, f3, f4 = self.forward_enc(x)
		output = self.forward_dec(f1, f2, f3, f4)
		return output
