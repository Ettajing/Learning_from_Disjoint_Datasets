import numpy as np
import pickle
import random
import os
from torchvision import transforms as T
import torch

class PatchPool(object):
	'''
	pool of patches, each patch contains a patch(n<10) index of a usr	
	db list of [[array(idx1, idx2), array(idx1, idx2)], ]

	Input: db_label -[ [usr1 label list], [usr2 label list], ...]
			fixed - update patch pools(train) or not (valid)
	content: self.db - [ n1, n2, n3, ..], n1 # of images for usr1
			self.pool - [patches of image pair]
	'''
	def __init__(self, db, patch_size=10):
		self.pool = []
		self.patch_size = patch_size
		self.db = db
		self.reset()

	def reset(self):
		# update patch pools
		self.pool = []
		for usr, data in enumerate(self.db):
			if len(data)>self.patch_size:
				data = np.random.permutation(data)
				while len(data)>self.patch_size:
					self.pool.append([usr, data[:self.patch_size]])
					data = data[self.patch_size:]

			if len(data)>0:
				self.pool.append([usr, data])

		random.shuffle(self.pool) 

	def randpop(self):
		if len(self.pool)==0:
			self.reset()
		i = np.random.randint(len(self.pool))
		data = self.pool.pop(i)
		return data

	def query(self, index):
		return self.pool[index]

class MyDataLoader(object):
	'''
	define my own dataloader, single worker
	'''
	def __init__(self, dbname, data_dir, split, id_offset=0, flip_tag=False, n_illu=5, n_pose=5, batch_size=100, patch_size=10, label_reorder=False):
		self.base_dir = data_dir
		self.dbname = dbname
		self.flip_tag = flip_tag
		self.batch_size = batch_size
		self.patch_size = patch_size
		self.id_offset = id_offset
		self.split = split
		self.illu_n = n_illu
		self.pose_n = n_pose
		self.label_reorder= label_reorder
		# sample indices
		self.label_list = []
		self.valid_list = []
		self.train_pair_list = []
		self.valid_pair_list = []
		self.load_label_list()
		self.count_db()

		# build pool
		self.POOL = PatchPool(self.train_pair_list, self.patch_size)
		self.batch = [[] for i in range(4)]

	def __len__(self):
		return self.n_img

	def load_label_list(self):
		if self.label_reorder:
			f = os.path.join(self.base_dir, 'data_info', self.dbname+'_label_list_order.txt')
		else:
			f = os.path.join(self.base_dir, 'data_info', self.dbname+'_label_list.txt')
			if not os.path.exists(f):
				f = os.path.join(self.base_dir, 'data_info', self.dbname+'_label_list0.txt')
		ll = pickle.load(open(f, 'rb'))
		self.label_list = ll[self.split[0]:self.split[1]]
		if self.split[-1]>len(ll): self.split[-1]==len(ll)
		self.valid_list = ll[self.split[-2]:self.split[-1]]

		if self.label_reorder:
			f = os.path.join(self.base_dir, 'data_info',self.dbname+'_pair_list_order.txt')
		else:
			f = os.path.join(self.base_dir, 'data_info',self.dbname+'_pair_list.txt')
			if not os.path.exists(f):
				f = os.path.join(self.base_dir, 'data_info',self.dbname+'_pair_list0.txt')

		ll = pickle.load(open(f, 'rb'))
		train_ll = ll[self.split[0]:self.split[1]]

		self.n_pair = 0
		if self.flip_tag:
			self.train_pair_list = []
			for usr in train_ll:
				N = len(usr)
				self.n_pair += N*2
				patch_f = np.zeros(N, dtype= np.int16)[:,np.newaxis]
				patch_o = np.ones(N, dtype= np.int16)[:, np.newaxis]
				self.train_pair_list.append(np.concatenate([np.concatenate([usr, patch_f], 1), np.concatenate([usr, patch_o], 1)], 0))
		else:
			self.train_pair_list = train_ll
			for i in train_ll:
				self.n_pair += len(i)

		self.valid_pair_list = ll[self.split[-2]:self.split[-1]]
		self.valid_batch = []
		cc = 0
		low_i = 0
		for i, ll in enumerate(self.valid_pair_list):
			cc += len(ll)
			if cc>self.batch_size:
				cc = 0
				self.valid_batch.append([low_i, i])
				low_i = i+1
		if low_i<i:
			self.valid_batch.append([low_i, i])

	def count_db(self):
		self.n_img = 0
		for usr in self.label_list:
			self.n_img += len(usr)
		print('total images of {}:{}'.format(self.dbname, self.n_img))
		print('total image pairs of {}:{}'.format(self.dbname, self.n_pair))
		print('total test batch', len(self.valid_batch))
	
	def max_id(self):
		max_id = 0
		for usr in self.label_list:
			if usr[0,0]>max_id:
				max_id = usr[0,0]
		return max_id

	def transform_batch(self, batch_img):
		batch_img = batch_img/255.0*2 - 1
		x = torch.Tensor(batch_img)
		return x
		
	def get_patch(self, data, split='train'): 
		# extract img of patch of each usr
		'''
			return an image array of a subject
			if self.mode=='single':
				return img, label
			else:
				return input_img, input_label, target_img, target_label

			img: n*c*imsize*imsize    np.array uint8
			label: n*4   np.array 
		'''
		usr, info = data
		flip = self.flip_tag
		if split=='train':
			label_list = self.label_list
		else:
			label_list = self.valid_list
			flip = False
			
		ID = label_list[usr][0, 0]
		f = os.path.join(self.base_dir, self.dbname, 'usr_'+str(ID)+'.txt')
		img = pickle.load(open(f, 'rb'))
		img = np.array(img, dtype=np.uint8)
		if len(img.shape)==3:
			img = img[:, np.newaxis]

		if len(info.shape)==1:  #single, non flip, valid
			in_img = img[info]
			in_label = label_list[usr][info]
			# in_label[;,0] += self.id_offset
			in_label = np.array(in_label, dtype=np.int16)
			return [in_img, in_label]

		elif info.shape[1]==2 and not flip: # pair, non flip
			in_indice = info[:, 0]
			out_indice = info[:, 1]

			in_img = img[in_indice]
			out_img = img[out_indice]

			in_label = label_list[usr][in_indice]
			out_label = label_list[usr][out_indice]

			# revise ID for different db
			in_label[:, 0] += self.id_offset
			out_label[:, 0] += self.id_offset
			in_label = np.array(in_label, dtype=np.int16)
			out_label = np.array(out_label, dtype= np.int16)
			return [in_img, in_label, out_img, out_label]
		else:  # pair, flip
			mask = (info[:,-1]==0) 
			ind_org = np.where(mask)[0]
			ind_flip = np.where(np.logical_not(mask))[0]

			in_img = np.zeros((len(info), img.shape[1], img.shape[2], img.shape[3]))
			in_label = np.zeros((len(info), label_list[usr].shape[1]))
			out_img = np.zeros((len(info), img.shape[1], img.shape[2], img.shape[3]))
			out_label = np.zeros((len(info), label_list[usr].shape[1]))

			# org
			in_indice = info[ind_org, 0]
			out_indice = info[ind_org, 1]
			in_img[ind_org] = img[in_indice]
			out_img[ind_org] = img[out_indice]
			in_label[ind_org] = label_list[usr][in_indice]
			out_label[ind_org] = label_list[usr][out_indice]

			flip_arr = np.zeros(len(info))
			# flip
			if len(ind_flip)>0:
				in_indice, out_indice = info[ind_flip, 0], info[ind_flip, 1]
				in_img[ind_flip] = img[in_indice, :,:,::-1]
				out_img[ind_flip] = img[out_indice, :,:, ::-1]
				y = label_list[usr][in_indice]
				y[:, 1] = self.illu_n-1 - y[:, 1] 
				y[:, 2] = self.pose_n-1 - y[:, 2]
				in_label[ind_flip]= y
				y = label_list[usr][out_indice]
				y[:, 1] = self.illu_n-1 - y[:, 1]
				y[:, 2] = self.pose_n-1 - y[:, 2]
				out_label[ind_flip] = y
				flip_arr[ind_flip] = 1
			flip_arr = flip_arr[:, np.newaxis]

			# revise ID for different 
			in_label[:, 0] += self.id_offset
			out_label[:, 0] += self.id_offset
			in_label = np.concatenate([in_label, flip_arr], 1)
			out_label = np.concatenate([out_label, flip_arr], 1)
			in_label = np.array(in_label, dtype=np.int16)
			out_label = np.array(out_label, dtype=np.int16)
			return [in_img, in_label, out_img, out_label]

	def differIndex(self, y1, y2):
		y1 = y1.squeeze()
		y2 = y2.squeeze()
		batchsize = len(y1)
		out1 = np.arange(batchsize)[y1!=y2]
		out = np.arange(batchsize)[y1==y2]

		return np.append(out1, out, 0), len(out1)

	def get_batch(self):
		while len(self.batch[0])<self.batch_size:
			patch = self.POOL.randpop()
			patch = self.get_patch(patch)
			if len(self.batch[0])==0: # empty history
				self.batch = [patch[i] for i in range(len(patch))]
			else:
				for i in range(len(patch)):
					self.batch[i] = np.append(self.batch[i], patch[i], 0)

		#---------- resize batch-------------
		if len(self.batch[0])>self.batch_size:
			output = [self.batch[i][:self.batch_size] for i in range(len(self.batch))]
			self.batch = [self.batch[i][self.batch_size:] for i in range(len(self.batch))]
		else:
			output = self.batch
			self.batch = [[]]

		return output
		
	def get_train_batch(self, check=0):
		output = self.get_batch()
		if check==1:
			valid = len(set(output[1][:,0]))>1 and len(set(output[1][:,1]))>1 and len(set(output[1][:,2]))>1
			while not valid:
				print('ID', len(set(output[1][:,0])), 'illu', len(set(output[1][:,1])), 'pose', len(set(output[1][:,2])))
				output = self.get_batch()
				valid = len(set(output[1][:,0]))>1 and len(set(output[1][:,1]))>1 and len(set(output[1][:,2]))>1
		elif check==2:
			valid = len(set(output[1][:,1]))==6 and len(set(output[1][:,2]))==7
			while not valid:
				print('ID', len(set(output[1][:,0])), 'illu', len(set(output[1][:,1])), 'pose', len(set(output[1][:,2])))
				output = self.get_batch()
				valid = len(set(output[1][:,1]))==6 and len(set(output[1][:,2]))==7 
		
		#print(output[1][:10,:], output[3][:10,:])
		return self.transform_batch(output[0]), output[1], self.transform_batch(output[2]), output[3]

	#===========valid db==============
	def get_fixed(self, n_per=10, n_usr=1):
		output = []
		for usr in range(n_usr):
			info = np.arange(len(self.valid_list[usr]))
			if len(info)>n_per:
				info = np.random.permutation(info)[:n_per]
			out = self.get_patch([usr, info], 'valid')
			if len(output)==0:
				output = out
			else:
				output = [np.concatenate([output[i], out[i]],0) for i in range(len(out))]
		return self.transform_batch(output[0]), output[1]

	def get_test_batch(self, ind):
		if ind>len(self.valid_batch)-1:
			print('warning: the index overflow the number of test batches')
			ind = 0

		output = []
		minn ,maxx = self.valid_batch[ind]
		for i in range(minn, maxx+1):
			patch = self.get_patch([i, self.valid_pair_list[i]], 'valid')
			if len(output)==0:
				output = patch
			else:
				for i in range(len(patch)):
					output[i] = np.append(output[i], patch[i], 0)

		#print(output[1][:10,:], output[3][:10,:])
		return self.transform_batch(output[0]), output[1], self.transform_batch(output[2]), output[3]
	
	def collect_all_images(self, mode='valid'):
		all_imgs, all_labels =[], []
		label_list = self.label_list if mode=='train' else self.valid_list

		for usr_ll in label_list:
			ID = usr_ll[0, 0]
			f = os.path.join(self.base_dir, self.dbname, 'usr_'+str(ID)+'.txt')
			img = pickle.load(open(f, 'rb'))
			img = np.array(img, dtype=np.uint8)
			if len(img.shape)==3:
				img = img[:, np.newaxis]
			all_imgs.append(img)
			all_labels.append(usr_ll)
		
		all_imgs = np.concatenate(all_imgs, 0)
		all_labels = np.concatenate(all_labels, 0)
	
		return self.transform_batch(all_imgs), all_labels
