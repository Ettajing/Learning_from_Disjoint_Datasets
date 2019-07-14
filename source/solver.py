from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from share.dataset import MyDataLoader
from share.losses import OnlineTripletLoss
from share.utils import RandomNegativeTripletSelector
from share.image_pool import ImagePool
import importlib


class Solver(object):
	"""Solver for training and testing StarGAN."""

	def __init__(self, config):
		"""Initialize configurations."""

		self.train_type = config.train_type
		self.D1 = config.D1
		self.D2 = config.D2
		self.D2_weight = config.D2_weight
		self.use_sigmoid = config.use_sigmoid
		self.use_pool = config.use_pool
		self.use_cyc = config.use_cyc
		self.use_triplet = config.use_triplet
		if self.use_pool: self.fake_pool = ImagePool(config.batch_size*3)

		self.net_gen_name = config.net_gen_name
		self.net_disc_name = config.net_disc_name
		self.gen_norm = config.gen_norm
		self.g_conv_dim = config.g_conv_dim
		self.d_conv_dim = config.d_conv_dim
		self.dim_attr = config.dim_attr
		self.dim_id = config.dim_id
		self.dim_source = config.dim_source
		self.attr_alpha = config.attr_alpha
		self.source_alpha = config.source_alpha
		self.id_alpha = config.id_alpha

		self.lambda_triplet = config.lambda_triplet
		self.lambda_map = config.lambda_map
		self.lambda_cyc = config.lambda_cyc
		self.lambda_cyc_id = config.lambda_cyc_id

		self.lambda_disc = config.lambda_disc
		self.lambda_D2 = config.lambda_D2
		self.lambda_D2_coef = config.lambda_D2_coef
		self.lambda_D2_coef_decay = config.lambda_D2_coef_decay

		self.data_dir = config.data_dir
		self.datasets = config.datasets 
		self.splits = config.splits 
		self.n_illu = config.n_illu 
		self.n_pose = config.n_pose
		self.image_c = config.image_c
		self.image_size = 128

		self.batch_size = config.batch_size
		self.patch_size = config.patch_size
		self.num_epoch = config.num_epoch
		self.g_lr = config.g_lr
		self.t_lr = config.t_lr
		self.g_lr_decay = config.g_lr_decay
		self.t_lr_decay = config.t_lr_decay
		self.d_lr = config.d_lr
		self.d_lr_decay = config.d_lr_decay
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.resume = config.resume
		self.log_dir = config.log_dir
		self.sample_dir = config.sample_dir
		self.model_save_dir = config.model_save_dir
		self.networks_dir = config.networks_dir

		# Step size.
		self.log_step = config.log_step 
		self.sample_step = config.sample_step
		self.model_save_step = config.model_save_step 
		self.lr_update_step = config.lr_update_step
		
		####################################################################################
		# Init dataloader
		self.prep_dataloader()
		self.test_batch()
		self.num_iter = self.n_samples//(self.batch_size*3)
		print('num_iter', self.num_iter)
		self.build_model()
		if self.D2 and self.D2_weight: self.get_D2_weight()

		# loss function
		self.loss_rec_fn = F.smooth_l1_loss
		#self.loss_rec_fn = F.l1_loss
		self.loss_map_fn = F.mse_loss
		self.loss_attr_triplet_fn = OnlineTripletLoss(self.attr_alpha, RandomNegativeTripletSelector(self.attr_alpha))
		self.loss_id_triplet_fn = OnlineTripletLoss(self.id_alpha, RandomNegativeTripletSelector(self.id_alpha))
		
		# save config
		self.current_time = time.strftime('%m-%d-%H-%M', time.gmtime())
		self.log_fname = os.path.join(self.log_dir, 'log-'+self.current_time+'.txt')
		self.save_dict(config, os.path.join(self.log_fname))

	def prep_dataloader(self):
		# Data loader.
		id_offset = 0
		self.n_samples = 0 
		self.attrs = ['illu', 'pose', 'both'] 
		self.dataloaders = []
		for i, db in enumerate(self.datasets):
			dataloader = MyDataLoader(db, self.data_dir, self.splits[i], id_offset, True, self.n_illu, self.n_pose, self.batch_size)
			id_offset += dataloader.max_id()
			self.n_samples += dataloader.n_pair
			self.dataloaders.append(dataloader)
			
	def build_model(self):
		net_mod = importlib.import_module('models.'+self.net_gen_name)
		self.G = net_mod.Generator(self.image_c, self.g_conv_dim, self.dim_attr, self.dim_id, self.dim_source,self.gen_norm)

		from models.model_map import Manipulator
		self.Map_illu = Manipulator(self.n_illu, self.dim_attr)
		self.Map_pose = Manipulator(self.n_pose, self.dim_attr)

		self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr)
		self.Map_illu_optimizer = torch.optim.Adam(self.Map_illu.parameters(), self.t_lr)
		self.Map_pose_optimizer = torch.optim.Adam(self.Map_pose.parameters(), self.t_lr)

		self.print_network(self.G, 'generator')
		self.print_network(self.Map_illu, 'T_illu')
		self.print_network(self.Map_pose, 'T_pose')

		self.G.to(self.device)
		self.Map_illu.to(self.device)
		self.Map_pose.to(self.device)

		if self.resume:
			dd = self.resume if os.path.exists(self.resume) else os.path.join(self.networks_dir, self.resume)
			self.restore_model(dd)

		if self.D1 or self.D2: # load disc
			disc_mod = importlib.import_module('models.'+self.net_disc_name)
			head = [self.use_sigmoid] if self.D1 else []
			if self.D2: head.extend([self.n_illu*2, self.n_pose*2])			
			self.Disc = disc_mod.Discriminator(self.image_c, self.d_conv_dim, head)
			self.disc_optimizer = torch.optim.Adam(self.Disc.parameters(), self.d_lr)
			self.print_network(self.Disc, 'Disc')
			self.Disc.to(self.device)

		self.G_state_t = 0
		self.D_state_t = 0
		self.M_state_t = 0

	def save_dict(self, log, savefile):
		with open(savefile, 'w') as out:
			for item in vars(log):
				out.write(item + ':'+ str(getattr(log, item))+'\n')

	def save_log(self, line, savefile):
		with open(savefile, 'a') as out:
			out.write(line+'\n')

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		#print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def reset_grad(self):
		self.G_optimizer.zero_grad()
		self.Map_pose_optimizer.zero_grad()
		self.Map_illu_optimizer.zero_grad()
		if self.D1 or self.D2:
			self.disc_optimizer.zero_grad()

	def restore_model(self, old_dir):
		"""Restore the trained generator and discriminator."""
		print('Loading the trained models from model dir {}...'.format(old_dir))
		G_path = os.path.join(old_dir, 'models', 'G.ckpt')
		Map_illu_path = os.path.join(old_dir, 'models', 'Map_illu.ckpt')
		Map_pose_path = os.path.join(old_dir, 'models', 'Map_pose.ckpt')

		self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
		self.Map_illu.load_state_dict(torch.load(Map_illu_path, map_location=lambda storage, loc: storage))
		self.Map_pose.load_state_dict(torch.load(Map_pose_path, map_location=lambda storage, loc: storage))

	def update_lr_G(self):
		"""Decay learning rates of the generator and discriminator."""
		g_lr = self.g_lr/(1+ self.G_state_t*self.g_lr_decay)
		for param_group in self.G_optimizer.param_groups:
			param_group['lr'] = g_lr

		self.G_state_t += 1

	def update_lr_M(self):
		t_lr = self.t_lr/(1+self.M_state_t* self.t_lr_decay)
		for param_group in self.Map_illu_optimizer.param_groups:
			param_group['lr'] = t_lr
		for param_group in self.Map_pose_optimizer.param_groups:
			param_group['lr'] = t_lr
		self.M_state_t +=1

	def update_lr_D(self):
		d_lr = self.d_lr/(1+self.D_state_t*self.d_lr_decay)
		for param_group in self.disc_optimizer.param_groups:
			param_group['lr'] = d_lr

		self.D_state_t += 1

	def denorm(self, x):
		"""Convert the range from [-1, 1] to [0, 1]."""
		out = (x + 1) / 2
		return out.clamp_(0, 1)
		
	def label2onehot(self, n_class, y):
		t = np.array(y)
		n = len(t)
		out = torch.zeros(n, n_class)
		out[np.arange(n), t] = 1
		return out

	def save_model(self):
		G_path = os.path.join(self.model_save_dir, 'G.ckpt')
		M_i_path = os.path.join(self.model_save_dir, 'Map_illu.ckpt')
		M_p_path = os.path.join(self.model_save_dir, 'Map_pose.ckpt')
		torch.save(self.G.state_dict(), G_path)
		torch.save(self.Map_illu.state_dict(), M_i_path)
		torch.save(self.Map_pose.state_dict(), M_p_path)

		if self.D1 or self.D2:
			torch.save(self.Disc.state_dict(), os.path.join(self.model_save_dir, 'D.ckpt'))
		print('Saved model checkpoints into {}...'.format(self.model_save_dir))

	def test_batch(self):
		test_x, test_y =[], []
		for dataloader in self.dataloaders:
			x, y = dataloader.get_fixed()
			test_x.append(x)
			test_y.append(y)
		if len(test_x)>0:
			self.test_x = torch.cat(test_x, 0).to(self.device)
			self.test_y = np.concatenate(test_y, 0)
		else:
			self.test_x = test_x[0].to(self.device)
			self.test_y = test_y[0]

	def test_visual(self, epoch):
		with torch.no_grad():
			n = self.test_x.size(0)
			x_fake_list = [self.test_x]
			illu_f, pose_f, id_f, source_f = self.G.forward_enc(self.test_x)
			rot = torch.zeros(n, self.n_illu)
			for k in range(self.n_illu):
				rot[:] = 0
				rot[:,k]=1
				fake_illu = self.Map_illu(illu_f, rot.to(self.device))
				x_fake = self.G.forward_dec(fake_illu, pose_f, id_f, source_f).detach()
				x_fake_list.append(x_fake)

			rot = torch.zeros(n, self.n_pose)
			for k in range(self.n_pose):
				rot[:] = 0
				rot[:,k]=1
				fake_pose = self.Map_pose(pose_f, rot.to(self.device))
				x_fake = self.G.forward_dec(illu_f, fake_pose, id_f, source_f).detach()
				x_fake_list.append(x_fake)

			x_rec = self.G.forward_dec(illu_f, pose_f, id_f, source_f).detach().cpu()
			x_concat = torch.cat(x_fake_list, dim=3).cpu()
			# revise non transform as rec
			for i in range(n):
				il, p = self.test_y[i, 1:3] 
				x_concat[i, :,:, self.image_size*(il+1): self.image_size*(il+2)] = x_rec[i]
				x_concat[i, :,:, self.image_size*(p+1+self.n_illu): self.image_size*(p+2+self.n_illu)] = x_rec[i]

			sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(epoch))
			save_image(self.denorm(x_concat), sample_path, nrow=1, padding=0)
			print('Saved real and fake images into {}...'.format(sample_path))

	def test_valid(self):
		# only cares about the rec loss
		with torch.no_grad():
			loss_valid = [0 for i in range(3)]
			for j, dataloader in enumerate(self.dataloaders):
				n = len(dataloader.valid_batch)
				for i in range(n):
					xx_img, __, tt_img, tt_label = dataloader.get_test_batch(i)
					feat_illu, feat_pose, feat_id, feat_source = self.G.forward_enc(xx_img.to(self.device))
					if self.attrs[j]=='illu' or self.attrs[j]=='both':
						rot_illu = self.label2onehot(self.n_illu, tt_label[:,1]).to(self.device)
						feat_illu = self.Map_illu(feat_illu, rot_illu)
					if self.attrs[j] =='pose' or self.attrs[j]=='both':
						rot_pose = self.label2onehot(self.n_pose, tt_label[:,2]).to(self.device)
						feat_pose = self.Map_pose(feat_pose, rot_pose)
					
					rec_tt = self.G.forward_dec(feat_illu, feat_pose, feat_id, feat_source)
					loss = self.loss_rec_fn(rec_tt, tt_img.to(self.device))
					loss_valid[j] =  loss_valid[j] + loss.item()
				loss_valid[j] = loss_valid[j]/n
		return loss_valid

	def gradient_penalty(self, y, x):
		"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
		weight = torch.ones(y.size()).to(self.device)
		dydx = torch.autograd.grad(outputs=y,
								   inputs=x,
								   grad_outputs=weight,
								   retain_graph=True,
								   create_graph=True,
								   only_inputs=True)[0]

		dydx = dydx.view(dydx.size(0), -1)
		dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
		return torch.mean((dydx_l2norm-1)**2)

	def get_D2_weight(self):
		# real [self.A_x_img, self.B_x_img, self.C_x_img]
		v = 2.0/(3*self.n_illu)
		weight_illu_real = [v for i in range(self.n_illu)]
		weight_illu_real[self.n_illu//2] += 1/3.0
		self.weight_illu_real = torch.Tensor(weight_illu_real).to(self.device)
		v = 2.0/(3*self.n_pose)
		weight_pose_real = [v for i in range(self.n_pose)]
		weight_pose_real[self.n_pose//2] += 1/3.0
		self.weight_pose_real = torch.Tensor(weight_pose_real).to(self.device)

		# false
		# [0.5self.A_y_f, 0.5self.A_y_t, 0.5self.B_y_f, 0.5self.B_y_t, self.C_y_t]
		m_illu_A = [1 for i in range(self.n_illu)]
		m_illu_B = [0.5 for i in range(self.n_illu)]
		m_illu_B[self.n_illu//2] += self.n_illu/2.0 # frontal
		weight_illu_fake = np.array([m_illu_A, m_illu_B, m_illu_A]).sum(0)
		weight_illu_fake /= (3*self.n_illu)
		self.weight_illu_fake = torch.Tensor(weight_illu_fake).to(self.device)

		m_pose_B = [1 for i in range(self.n_pose)]
		m_pose_A = [0.5 for i in range(self.n_pose)]
		m_pose_A[self.n_pose//2] += self.n_pose/2.0 # frontal
		weight_pose_fake = np.array([m_pose_A, m_pose_B, m_pose_A]).sum(0)
		weight_pose_fake /= (3*self.n_pose)
		self.weight_pose_fake = torch.Tensor(weight_pose_fake).to(self.device)

		self.pad_il = torch.tensor(0.0).expand_as(self.weight_illu_real).to(self.device)
		self.pad_p = torch.tensor(0.0).expand_as(self.weight_pose_real).to(self.device)

		self.weight_illu_real = torch.cat([self.weight_illu_real, self.pad_il])
		self.weight_pose_real = torch.cat([self.weight_pose_real, self.pad_p])

	def update_disc(self):
		# real images
		real_img = torch.cat([self.A_x_img, self.B_x_img, self.C_x_img], 0)
		real_label = np.concatenate([self.A_x_label, self.B_x_label, self.C_x_label], 0)

		# fake images
		ind = torch.randperm(self.batch_size)
		ind_f = ind[:self.batch_size//2]
		ind_t = ind[self.batch_size//2:]
		if not self.use_pool:
			fake_img = torch.cat([self.A_y_f[ind_f].detach(), self.A_y_t[ind_t].detach(), self.B_y_f[ind_f].detach(), self.B_y_t[ind_t].detach(), self.C_y_t.detach()], 0)
			fake_label = np.concatenate([self.A_fake_label[ind_f], self.A_t_label[ind_t, :3], self.B_fake_label[ind_f], self.B_t_label[ind_t,:3], self.C_t_label[:,:3]], 0)
		else:
			fakeA1, fake_lA1 = self.fake_pool.query(self.A_y_f[ind_f].detach(), self.A_fake_label[ind_f])
			fakeA2, fake_lA2 = self.fake_pool.query(self.A_y_t[ind_t].detach(), self.A_t_label[ind_t, :3])
			fakeB1, fake_lB1 = self.fake_pool.query(self.B_y_f[ind_f].detach(), self.B_fake_label[ind_f])
			fakeB2, fake_lB2 = self.fake_pool.query(self.B_y_t[ind_t].detach(), self.B_t_label[ind_t, :3])
			fakeC, fake_lC = self.fake_pool.query(self.C_y_t.detach(), self.C_t_label[:, :3])
			fake_img = torch.cat([fakeA1, fakeA2, fakeB1, fakeB2, fakeC], 0)
			fake_label = np.concatenate([fake_lA1, fake_lA2, fake_lB1, fake_lB2, fake_lC], 0)

		# discriminator loss 
		pred_real = self.Disc(real_img)
		pred_fake = self.Disc(fake_img)

		v_D1_real, v_D1_fake, v_D2_il, v_D2_p = 0, 0, 0, 0
		if self.D1:
			if self.use_sigmoid: #bce
				target_d1_real = torch.tensor(1.0).expand_as(pred_real[0]).to(self.device)
				target_d1_fake = torch.tensor(0.0).expand_as(pred_fake[0]).to(self.device)
				loss_D1_real = F.binary_cross_entropy(pred_real[0], target_d1_real)
				loss_D1_fake = F.binary_cross_entropy(pred_fake[0], target_d1_fake)
				loss_D1 = loss_D1_real + loss_D1_fake
			else: # wgan
				loss_D1_real = -torch.mean(pred_real[0])
				loss_D1_fake =  torch.mean(pred_fake[0])

				alpha = torch.rand(len(real_img), 1,1,1).to(self.device)
				input_hat = (alpha* real_img + (1 - alpha) * fake_img).requires_grad_(True)
				pred_hat = self.Disc(input_hat)
				loss_gp = self.gradient_penalty(pred_hat[0], input_hat)
				loss_D1 = loss_D1_real + loss_D1_fake + loss_gp
			v_D1_real, v_D1_fake = loss_D1_real.item(), loss_D1_fake.item()
		if self.D2:
			if self.D1:
				pred_il_real, pred_p_real = pred_real[1], pred_real[2]
				pred_il_fake, pred_p_fake = pred_fake[1], pred_fake[2]
			else:
				pred_il_real, pred_p_real = pred_real[0], pred_real[1]
				pred_il_fake, pred_p_fake = pred_fake[0], pred_fake[1]

			target_D2_real = torch.LongTensor(real_label[:, 1:3]).to(self.device)
			target_D2_fake = torch.LongTensor(fake_label[:, 1:3]).add(torch.LongTensor([self.n_illu, self.n_pose])).to(self.device) 

			lambda_D2_coef = self.lambda_D2_coef*(1- self.D_state_t*self.lambda_D2_coef_decay)
			if not self.D2_weight:
				loss_D2_il, loss_D2_p = F.cross_entropy(pred_il_real, target_D2_real[:, 0]), F.cross_entropy(pred_p_real, target_D2_real[:, 1])
				if lambda_D2_coef>0:
					loss_D2_il_f, loss_D2_p_f = F.cross_entropy(pred_il_fake, target_D2_fake[:, 0]), F.cross_entropy(pred_p_fake, target_D2_fake[:, 1])
			else:
				loss_D2_il = F.cross_entropy(pred_il_real, target_D2_real[:, 0], weight= self.weight_illu_real)
				loss_D2_p = F.cross_entropy(pred_p_real, target_D2_real[:, 1], weight = self.weight_pose_real)
				if lambda_D2_coef>0:
					loss_D2_il_f = F.cross_entropy(pred_il_fake, target_D2_fake[:, 0], weight = torch.cat([self.pad_il, self.weight_illu_fake]))
					loss_D2_p_f = F.cross_entropy(pred_p_fake, target_D2_fake[:, 1], weight= torch.cat([self.pad_p, self.weight_pose_fake]))
			
			if lambda_D2_coef>0:
				loss_D2_il = loss_D2_il + loss_D2_il_f*lambda_D2_coef
				loss_D2_p = loss_D2_p + loss_D2_p_f*lambda_D2_coef
			v_D2_il, v_D2_p = loss_D2_il.item(), loss_D2_p.item()

		if self.D1:
			loss_D = loss_D1 
			if self.D2: loss_D = loss_D + self.lambda_D2*(loss_D2_il + loss_D2_p)
		elif self.D2:
			loss_D = loss_D2_il + loss_D2_p

		loss_D.backward()
		self.disc_optimizer.step()

		return v_D1_real, v_D1_fake, v_D2_il, v_D2_p

	def train_0(self, num_epoch):
		"""Train face generator with manipulator on C 
		"""
		# Start training.
		print('Start training...')
		start_time = time.time()
		for i in range(num_epoch):
			for j in range(self.num_iter):
				self.reset_grad()
				self.A_x_img, self.A_x_label, self.A_t_img, self.A_t_label = self.dataloaders[0].get_train_batch()
				self.A_x_img, self.A_t_img = self.A_x_img.to(self.device), self.A_t_img.to(self.device)
				self.B_x_img, self.B_x_label, self.B_t_img, self.B_t_label= self.dataloaders[1].get_train_batch() 
				self.B_x_img, self.B_t_img = self.B_x_img.to(self.device), self.B_t_img.to(self.device)
				A_t_illu, A_t_pose, A_t_id, A_t_source = self.G.forward_enc(self.A_t_img)
				A_x_illu, A_x_pose, A_x_id, A_x_source = self.G.forward_enc(self.A_x_img)
				B_t_illu, B_t_pose, B_t_id, B_t_source = self.G.forward_enc(self.B_t_img)
				B_x_illu, B_x_pose, B_x_id, B_x_source = self.G.forward_enc(self.B_x_img)
		
				A_rot_illu = self.label2onehot(self.n_illu, self.A_t_label[:, 1]).to(self.device)
				A_x_illu_t = self.Map_illu(A_x_illu, A_rot_illu)
				self.A_y_t = self.G.forward_dec(A_x_illu_t, A_x_pose, A_x_id, A_x_source)
				
				B_rot_pose = self.label2onehot(self.n_pose, self.B_t_label[:, 2]).to(self.device)
				B_x_pose_t = self.Map_pose(B_x_pose, B_rot_pose)
				self.B_y_t = self.G.forward_dec(B_x_illu, B_x_pose_t, B_x_id, B_x_source)

				self.A_fake_label = np.concatenate([self.A_x_label[:,:2], self.B_t_label[:, 2:3]], 1)
				A_x_pose_f = self.Map_pose(A_x_pose, B_rot_pose)

				self.B_fake_label = np.concatenate([self.B_x_label[:, 0:1], self.A_t_label[:,1:2], self.B_x_label[:, 2:3]], 1)
				B_x_illu_f = self.Map_illu(B_x_illu, A_rot_illu)

				loss_map_illu_A = self.loss_map_fn(A_x_illu_t, A_t_illu)
				loss_xt_A = self.loss_map_fn(A_x_pose, A_t_pose) + self.loss_map_fn(A_x_id, A_t_id)
				loss_map_pose_A = self.loss_map_fn(A_x_pose_f, B_t_pose)
				loss_A_y = self.loss_rec_fn(self.A_y_t, self.A_t_img) 

				loss_map_pose_B = self.loss_map_fn(B_x_pose_t, B_t_pose)
				loss_xt_B = self.loss_map_fn(B_x_illu, B_t_illu) + self.loss_map_fn(B_x_id, B_t_id)
				loss_map_illu_B = self.loss_map_fn(B_x_illu_f, A_t_illu)
				loss_B_y = self.loss_rec_fn(self.B_y_t, self.B_t_img)

				# ========C==============
				self.C_x_img, self.C_x_label, self.C_t_img, self.C_t_label= self.dataloaders[2].get_train_batch() 
				self.C_x_img, self.C_t_img = self.C_x_img.to(self.device), self.C_t_img.to(self.device)
				C_t_illu, C_t_pose, C_t_id, C_t_source = self.G.forward_enc(self.C_t_img)
				C_x_illu, C_x_pose, C_x_id, C_x_source = self.G.forward_enc(self.C_x_img)
			
				C_rot_illu = self.label2onehot(self.n_illu, self.C_t_label[:, 1]).to(self.device)
				C_rot_pose = self.label2onehot(self.n_pose, self.C_t_label[:, 2]).to(self.device)
				C_x_illu_t = self.Map_illu(C_x_illu, C_rot_illu)
				C_x_pose_t = self.Map_pose(C_x_pose, C_rot_pose)
				self.C_y_t = self.G.forward_dec(C_x_illu_t, C_x_pose_t, C_x_id, C_x_source)

				loss_map_illu_C = self.loss_map_fn(C_x_illu_t, C_t_illu)
				loss_map_pose_C = self.loss_map_fn(C_x_pose_t, C_t_pose)
				loss_xt_C = self.loss_map_fn(C_x_id, C_t_id)
				loss_C_y = self.loss_rec_fn(self.C_y_t, self.C_t_img)

				#============= feature space =============
				# feat loss
				if self.use_triplet:
					real_label = np.concatenate([self.A_x_label, self.B_x_label, self.C_x_label], 0)
					real_illu = torch.cat([A_x_illu, B_x_illu, C_x_illu], 0)
					real_pose = torch.cat([A_x_pose, B_x_pose, C_x_pose], 0)
					real_id = torch.cat([A_x_id, B_x_id, C_x_id],0)
					loss_illu, __ = self.loss_attr_triplet_fn(real_illu, real_label[:,1])
					loss_pose, __ = self.loss_attr_triplet_fn(real_pose, real_label[:,2])
					loss_id, __ = self.loss_id_triplet_fn(real_id, real_label[:,0])
				
				## ========== total loss
				loss_total = loss_C_y +loss_A_y + loss_B_y +\
						(loss_map_illu_C + loss_map_pose_C + loss_map_illu_A + loss_map_illu_B + loss_map_pose_A + loss_map_pose_B + loss_xt_A + loss_xt_B + loss_xt_C) * self.lambda_map
				if self.use_triplet:
					loss_total = loss_total + (loss_illu + loss_pose + loss_id)*self.lambda_triplet
				loss_total.backward()
				self.G_optimizer.step()
				self.Map_pose_optimizer.step()
				self.Map_illu_optimizer.step()

				if (i*self.num_iter+j)% self.lr_update_step==0:
					self.update_lr_G()
					self.update_lr_M()

			ss = 'update g_lr:{:.8f}, t_lr:{:.8f}'.format(self.g_lr/(1+(self.G_state_t-1)*self.g_lr_decay), self.t_lr/(1+(self.M_state_t-1)*self.t_lr_decay))
			print(ss)
			# Print out training information.
			if (i+1) % self.log_step == 0:
				loss_valid_A, loss_valid_B, loss_valid_C = self.test_valid()

				loss = {}
				loss['G/rec_A_y'] = loss_A_y.item()
				loss['G/rec_B_y'] = loss_B_y.item()
				loss['G/rec_C_y'] = loss_C_y.item()
				if self.use_triplet:
					loss['F/illu'] = loss_illu.item()
					loss['F/pose'] = loss_pose.item()
					loss['F/id'] = loss_id.item()
			
				loss['M/A_pose'] = loss_map_pose_A.item()
				loss['M/A_illu'] = loss_map_illu_A.item()
				loss['M/B_pose'] = loss_map_pose_B.item()
				loss['M/B_illu'] = loss_map_illu_B.item()
				loss['M/C_pose'] = loss_map_pose_C.item()
				loss['M/C_illu'] = loss_map_illu_C.item()
				loss['V/A'] = loss_valid_A
				loss['V/B'] = loss_valid_B
				loss['V/C'] = loss_valid_C

				et = time.time() - start_time
				et = str(datetime.timedelta(seconds=et))[:-7]
				log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}]".format(et, i+1, num_epoch, (i+1)*self.num_iter)
				for tag, value in loss.items():
					log += ", {}: {:.6f}".format(tag, value)
				print(log)
				self.save_log(log, self.log_fname)
				
			# Translate fixed images for debugging.
			if (i+1) % self.sample_step == 0: self.test_visual(i+1)

			# Save model checkpoints.
			if (i+1) % self.model_save_step == 0: 
				self.save_model()
				self.save_log(ss, self.log_fname)

	def train_1(self):
		"""Train face generator with manipulator.
		"""
		# Start training.
		print('Start training...')
		start_time = time.time()
		for i in range(self.num_epoch):
			for j in range(self.num_iter):
				self.reset_grad()
				#===================A, illumination change with ground truth================
				self.A_x_img, self.A_x_label, self.A_t_img, self.A_t_label = self.dataloaders[0].get_train_batch()
				self.A_x_img, self.A_t_img = self.A_x_img.to(self.device), self.A_t_img.to(self.device)
				self.B_x_img, self.B_x_label, self.B_t_img, self.B_t_label= self.dataloaders[1].get_train_batch() 
				self.B_x_img, self.B_t_img = self.B_x_img.to(self.device), self.B_t_img.to(self.device)
				A_t_illu, A_t_pose, A_t_id, A_t_source = self.G.forward_enc(self.A_t_img)
				A_x_illu, A_x_pose, A_x_id, A_x_source = self.G.forward_enc(self.A_x_img)
				B_t_illu, B_t_pose, B_t_id, B_t_source = self.G.forward_enc(self.B_t_img)
				B_x_illu, B_x_pose, B_x_id, B_x_source = self.G.forward_enc(self.B_x_img)
		
				A_rot_illu = self.label2onehot(self.n_illu, self.A_t_label[:, 1]).to(self.device)
				A_x_illu_t = self.Map_illu(A_x_illu, A_rot_illu)
				self.A_y_t = self.G.forward_dec(A_x_illu_t.detach(), A_x_pose.detach(), A_x_id.detach(), A_x_source)
				
				B_rot_pose = self.label2onehot(self.n_pose, self.B_t_label[:, 2]).to(self.device)
				B_x_pose_t = self.Map_pose(B_x_pose, B_rot_pose)
				self.B_y_t = self.G.forward_dec(B_x_illu.detach(), B_x_pose_t.detach(), B_x_id.detach(), B_x_source)

				self.A_fake_label = np.concatenate([self.A_x_label[:,:2], self.B_t_label[:, 2:3]], 1)
				A_x_pose_f = self.Map_pose(A_x_pose, B_rot_pose)
				self.A_y_f = self.G.forward_dec(A_x_illu.detach(), A_x_pose_f.detach(), A_x_id.detach(), A_x_source)

				self.B_fake_label = np.concatenate([self.B_x_label[:, 0:1], self.A_t_label[:,1:2], self.B_x_label[:, 2:3]], 1)
				B_x_illu_f = self.Map_illu(B_x_illu, A_rot_illu)
				self.B_y_f = self.G.forward_dec(B_x_illu_f.detach(), B_x_pose.detach(), B_x_id.detach(), B_x_source)

				loss_map_illu_A = self.loss_map_fn(A_x_illu_t, A_t_illu)
				loss_xt_A = self.loss_map_fn(A_x_pose, A_t_pose) + self.loss_map_fn(A_x_id, A_t_id)
				loss_map_pose_A = self.loss_map_fn(A_x_pose_f, B_t_pose)
				loss_A_y = self.loss_rec_fn(self.G.forward_dec(A_x_illu_t, A_x_pose, A_x_id, A_x_source), self.A_t_img) 

				loss_map_pose_B = self.loss_map_fn(B_x_pose_t, B_t_pose)
				loss_xt_B = self.loss_map_fn(B_x_illu, B_t_illu) + self.loss_map_fn(B_x_id, B_t_id)
				loss_map_illu_B = self.loss_map_fn(B_x_illu_f, A_t_illu)
				loss_B_y = self.loss_rec_fn(self.G.forward_dec(B_x_illu, B_x_pose_t, B_x_id, B_x_source), self.B_t_img)

				# ========C joint dataset==============
				self.C_x_img, self.C_x_label, self.C_t_img, self.C_t_label= self.dataloaders[2].get_train_batch() 
				self.C_x_img, self.C_t_img = self.C_x_img.to(self.device), self.C_t_img.to(self.device)
				C_t_illu, C_t_pose, C_t_id, C_t_source = self.G.forward_enc(self.C_t_img)
				C_x_illu, C_x_pose, C_x_id, C_x_source = self.G.forward_enc(self.C_x_img)
			
				C_rot_illu = self.label2onehot(self.n_illu, self.C_t_label[:, 1]).to(self.device)
				C_rot_pose = self.label2onehot(self.n_pose, self.C_t_label[:, 2]).to(self.device)
				C_x_illu_t = self.Map_illu(C_x_illu, C_rot_illu)
				C_x_pose_t = self.Map_pose(C_x_pose, C_rot_pose)
				self.C_y_t = self.G.forward_dec(C_x_illu_t.detach(), C_x_pose_t.detach(), C_x_id.detach(), C_x_source)

				loss_map_illu_C = self.loss_map_fn(C_x_illu_t, C_t_illu)
				loss_map_pose_C = self.loss_map_fn(C_x_pose_t, C_t_pose)
				loss_xt_C = self.loss_map_fn(C_x_id, C_t_id)
				loss_C_y = self.loss_rec_fn(self.G.forward_dec(C_x_illu_t, C_x_pose_t, C_x_id, C_x_source), self.C_t_img) 

				#=====================Disc=========================
				if self.D1 or self.D2: loss_d1_real, loss_d1_fake, loss_d2_il, loss_d2_p = self.update_disc()
				#============= Gen=============
				loss_total = loss_A_y + loss_B_y + loss_C_y + \
							(loss_map_illu_A + loss_map_pose_B + loss_map_illu_C + loss_map_pose_C + loss_map_illu_B + loss_map_pose_A + loss_xt_B+ loss_xt_A + loss_xt_C) *self.lambda_map
				# feat loss
				if self.use_triplet:
					real_label = np.concatenate([self.A_x_label, self.B_x_label, self.C_x_label], 0)
					real_illu = torch.cat([A_x_illu, B_x_illu, C_x_illu], 0)
					real_pose = torch.cat([A_x_pose, B_x_pose, C_x_pose], 0)
					real_id = torch.cat([A_x_id, B_x_id, C_x_id],0)
					loss_illu, __ = self.loss_attr_triplet_fn(real_illu, real_label[:,1])
					loss_pose, __ = self.loss_attr_triplet_fn(real_pose, real_label[:,2])
					loss_id, __ = self.loss_id_triplet_fn(real_id, real_label[:,0])
					loss_total = loss_total + (loss_illu+ loss_pose+ loss_id)*self.lambda_triplet 
	
				# cycle 
				if self.use_cyc:
					A_yf_illu, A_yf_pose, A_yf_id, A_yf_source = self.G.forward_enc(self.A_y_f)
					A_yt_illu, A_yt_pose, A_yt_id, A_yt_source = self.G.forward_enc(self.A_y_t)
					B_yf_illu, B_yf_pose, B_yf_id, B_yf_source = self.G.forward_enc(self.B_y_f)
					B_yt_illu, B_yt_pose, B_yt_id, B_yt_source = self.G.forward_enc(self.B_y_t)
					C_yt_illu, C_yt_pose, C_yt_id, C_yt_source = self.G.forward_enc(self.C_y_t)

					loss_cyc_attr = (F.mse_loss(A_yf_illu, A_x_illu.detach()) + F.mse_loss(A_yf_pose, B_t_pose.detach()) + \
									F.mse_loss(A_yt_illu, A_t_illu.detach()) + F.mse_loss(A_yt_pose, A_x_pose.detach()) +\
									F.mse_loss(B_yf_illu, A_t_illu.detach()) + F.mse_loss(B_yf_pose, B_x_pose.detach())+\
									F.mse_loss(B_yt_illu, B_x_illu.detach()) + F.mse_loss(B_yt_illu, B_t_pose.detach()) + \
									F.mse_loss(C_yt_illu, C_t_illu.detach()) + F.mse_loss(C_yt_pose, C_t_pose.detach()))/10
					loss_cyc_id = (F.mse_loss(A_yf_id, A_x_id.detach()) + F.mse_loss(A_yt_id, A_t_id.detach()) + \
									F.mse_loss(B_yf_id, B_x_id.detach()) + F.mse_loss(B_yt_id, B_t_id.detach()) +\
									F.mse_loss(C_yt_id, C_x_id.detach()))/5
					loss_cyc = loss_cyc_attr + loss_cyc_id* self.lambda_cyc_id
					loss_total = loss_total + loss_cyc*self.lambda_cyc

				# disc loss
				if self.D1 or self.D2:
					# fake images
					ind = torch.randperm(self.batch_size)
					ind_f = ind[:self.batch_size//2]
					ind_t = ind[self.batch_size//2:]
					fake_img = torch.cat([self.A_y_f[ind_f], self.A_y_t[ind_t], self.B_y_f[ind_f], self.B_y_t[ind_t], self.C_y_t], 0)
					fake_label = np.concatenate([self.A_fake_label[ind_f], self.A_t_label[ind_t, :3], self.B_fake_label[ind_f], self.B_t_label[ind_t,:3], self.C_t_label[:,:3]], 0)
					target_D2_fake = torch.LongTensor(fake_label[:,1:3]).to(self.device)
					pred_fake = self.Disc(fake_img)
					if self.D1:
						if self.use_sigmoid:
							target_d1_real = torch.tensor(1.0).expand_as(pred_fake[0]).to(self.device)
							loss_D1 = F.binary_cross_entropy(pred_fake[0], target_d1_real)
						else:
							loss_D1 = -torch.mean(pred_fake[0])

					if self.D2:
						if self.D1:
							pred_illu, pred_pose = pred_fake[1], pred_fake[2]
						else:
							pred_illu, pred_pose = pred_fake[0], pred_fake[1]
						if self.D2_weight:
							loss_D2_il = F.cross_entropy(pred_illu, target_D2_fake[:,0], weight = torch.cat([self.weight_illu_fake, self.pad_il]))
							loss_D2_p = F.cross_entropy(pred_pose, target_D2_fake[:,1], weight= torch.cat([self.weight_pose_fake, self.pad_p]))
							loss_D2 = loss_D2_il + loss_D2_p
						else:
							loss_D2 = F.cross_entropy(pred_illu, target_D2_fake[:, 0]) + F.cross_entropy(pred_pose, target_D2_fake[:,1])

					if self.D1:
						loss_disc = loss_D1
						if self.D2: loss_disc = loss_disc + loss_D2*self.lambda_D2
					elif self.D2:
						loss_disc = loss_D2*self.lambda_D2

					loss_total = loss_total + loss_disc*self.lambda_disc
				
				loss_total.backward()
				self.G_optimizer.step()
				self.Map_pose_optimizer.step()
				self.Map_illu_optimizer.step()

				if (i*self.num_iter+ j)% self.lr_update_step ==0:
					self.update_lr_G()
					self.update_lr_M()
					self.update_lr_D()

			ss = 'update g_lr:{:.8f}, t_lr:{:.8f}, d_lr{:.8f}'.format(self.g_lr/(1+(self.G_state_t-1)*self.g_lr_decay), self.t_lr/(1+(self.M_state_t-1)*self.t_lr_decay), self.d_lr/(1+(self.D_state_t-1)*self.d_lr_decay))
			print(ss)		
			# Print out training information.
			if (i+1) % self.log_step == 0:
				loss_valid_A, loss_valid_B, loss_valid_C = self.test_valid()

				loss = {}
				if self.D1:
					loss['D1/real'] = loss_d1_real
					loss['D1/fake'] = loss_d1_fake
				if self.D2:
					loss['D2/illu'] = loss_d2_il
					loss['D2/pose'] = loss_d2_p

				loss['G/rec_C_y'] = loss_C_y.item()
				loss['G/rec_A_y'] = loss_A_y.item()
				loss['G/rec_B_y'] = loss_B_y.item()
				if self.D1 or self.D2: loss['GD'] = loss_disc.item()
				if self.use_triplet:
					loss['F/illu'] = loss_illu.item()
					loss['F/pose'] = loss_pose.item()
					loss['F/id'] = loss_id.item()
			
				loss['M/A_illu'] = loss_map_illu_A.item()
				loss['M/A_pose'] = loss_map_pose_A.item()
				loss['M/B_illu'] = loss_map_illu_B.item()
				loss['M/B_pose'] = loss_map_pose_B.item()
				loss['M/C_illu'] = loss_map_illu_C.item()
				loss['M/C_pose'] = loss_map_pose_C.item()
				
				if self.use_cyc:
					loss['cyc/fake_attr'] = loss_cyc_attr.item()
					loss['cyc/id'] = loss_cyc_id.item()
					#loss['cyc/source'] = loss_cyc_source.item()
				loss['V/A'] = loss_valid_A
				loss['V/B'] = loss_valid_B
				loss['V/C'] = loss_valid_C

				et = time.time() - start_time
				et = str(datetime.timedelta(seconds=et))[:-7]
				log = "Elapsed [{}], Epoch [{}/{}], Iteration [{}]".format(et, i+1, self.num_epoch, (i+1)*self.num_iter)
				for tag, value in loss.items():
					log += ", {}: {:.6f}".format(tag, value)
				print(log)
				self.save_log(log, self.log_fname)
				
			# Translate fixed images for debugging.
			if (i+1) % self.sample_step == 0: self.test_visual(i+1)

			# Save model checkpoints.
			if (i+1) % self.model_save_step == 0: 
				self.save_model()
				self.save_log(ss, self.log_fname)
			
	def train(self):
		loss_A, loss_B, loss_C = self.test_valid()
		print('loss_A', loss_A, 'loss_B', loss_B, 'loss_C', loss_C)
		if self.train_type==0:
			self.train_0(30)
		elif self.train_type==1:
			'''
			self.test_visual(0)

			self.G_state_t = (30 * self.num_iter)//self.lr_update_step
			self.M_state_t = self.G_state_t
			self.update_lr_G()
			self.update_lr_M()
			'''
			self.train_1()
