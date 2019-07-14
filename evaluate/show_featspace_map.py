import torch
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import random
from code import options_set2_1 as options
from models.model_map import Manipulator
import importlib

def tsne_map():
	Feat = []
	Feat_illu = []
	Feat_pose = []
	Label = []
	db_nums = []
	N = 1000
	for db_i, db in enumerate(config.datasets):
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f= f[:-4]+'0.txt'
		labels = pickle.load(open(f, 'rb'))[config.splits[db_i][1]:]
		c = 0
		for usr_label in labels:
			ID = usr_label[0][0]
			f = os.path.join(config.data_dir, db, 'usr_' + str(ID)+'.txt')
			img = pickle.load(open(f, 'rb'))
			img = np.array(img)
			if len(img.shape)<4: img = img[:, np.newaxis]
			img = img/255.0*2-1
			img = torch.Tensor(img).to(device)
			x_illu, x_pose, x_id, x_source = G.forward_enc(img)

			new_illu = []
			rot = torch.zeros(len(img), config.n_illu).to(device)
			for ii in range(config.n_illu):
				rot[:] = 0
				rot[:, ii] = 1
				x = Map_illu(x_illu, rot)
				new_illu.append(x.data.cpu().numpy())

			new_pose = []
			rot = torch.zeros(len(img), config.n_pose).to(device)
			for ii in range(config.n_pose):
				rot[:]=0
				rot[:, ii] = 1
				x = Map_pose(x_pose, rot)
				new_pose.append(x.data.cpu().numpy())

			if len(Label) ==0:
				Label = usr_label
				Feat_illu = new_illu
				Feat_pose = new_pose
				Feat = [x_illu.data.cpu().numpy(), x_pose.data.cpu().numpy()]
			else:
				Label = np.append(Label, usr_label, 0)
				Feat[0] = np.append(Feat[0], x_illu.data.cpu().numpy(), 0)
				Feat[1] = np.append(Feat[1], x_pose.data.cpu().numpy(), 0)
				for ii in range(config.n_illu):
					Feat_illu[ii] = np.append(Feat_illu[ii], new_illu[ii], 0)
				for ii in range(config.n_pose):
					Feat_pose[ii] = np.append(Feat_pose[ii], new_pose[ii], 0)

			c += len(usr_label)
			if c>=N: 
				db_nums.append(c)
				break
			
	# tsne show
	# illu
	print(db_nums, np.sum(np.array(db_nums)))
	print(len(Feat[0]))
	all_illu_feat = np.copy(Feat[0])
	label_illu = np.copy(Label[:, 1])
	temp_label = Label[:,1]
	for i in range(config.n_illu):
		indices = np.where(np.logical_not(temp_label==i+1))[0]
		if len(indices)==0: continue
		print(all_illu_feat.shape, Feat_illu[i].shape)
		all_illu_feat = np.append(all_illu_feat, Feat_illu[i][indices,:], 0)
		new_label = np.zeros(len(indices))
		new_label[:]= i
		label_illu = np.append(label_illu, new_label)

	all_illu_feat = TSNE(n_components=2).fit_transform(all_illu_feat)
	print(all_illu_feat.shape, label_illu.shape)

	all_pose_feat = np.copy(Feat[1])
	label_pose = np.copy(Label[:, 2])
	temp_label = Label[:,2]
	for i in range(config.n_pose):
		indices = np.where(np.logical_not(temp_label==i+1))[0]
		if len(indices)==0: continue
		all_pose_feat = np.append(all_pose_feat, Feat_pose[i][indices,:], 0)
		new_label = np.zeros(len(indices))
		new_label[:]= i
		label_pose = np.append(label_pose, new_label)

	all_pose_feat = TSNE(n_components=2).fit_transform(all_pose_feat)
	print(all_pose_feat.shape, label_pose.shape)

	result= {'Feat_illu': all_illu_feat, 'Label_illu': label_illu, 'Feat_pose': all_pose_feat, 'Label_pose': label_pose}
	with open(os.path.join(evaluate_dir, 'map_tsne_data.txt'), 'wb') as fp:
		pickle.dump(result, fp)

if __name__=='__main__':
	parser = options.TrainOptions()
	config = parser.parse()
	os.environ['CUDA_VISIBLE_DEVICES']=str(config.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = config.name
	print(model)
	gen_mod = importlib.import_module('models.'+config.net_gen_name)

	model_path = os.path.join(config.networks_dir, model, 'models')
	evaluate_dir = os.path.join(config.networks_dir, model, 'evaluate_featspace')
	if not os.path.exists(evaluate_dir): os.makedirs(evaluate_dir)

	# build model
	imgsize = 128
	G = gen_mod.Generator(config.image_c, config.g_conv_dim, config.dim_attr, config.dim_id, config.dim_source, config.gen_norm)
	G.to(device).eval()
	G.load_state_dict(torch.load(os.path.join(model_path, 'G.ckpt'), map_location=lambda storage, loc:storage))
	
	Map_illu = Manipulator(config.n_illu, config.dim_attr)
	Map_pose = Manipulator(config.n_pose, config.dim_attr)
	Map_illu.to(device)
	Map_pose.to(device)	
	Map_illu.load_state_dict(torch.load(os.path.join(model_path, 'Map_illu.ckpt'), map_location=lambda storage, loc:storage))
	Map_pose.load_state_dict(torch.load(os.path.join(model_path, 'Map_pose.ckpt'), map_location=lambda storage, loc:storage))
	#----- classification---------

	tsne_map()
