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
	# load data from PIE, CAS, NCKU
	Feat = [[] for i in range(config.n_illu+config.n_pose)]
	Label = []
	N = [1000, 500, 2000]
	for db_i, db in enumerate(config.datasets):
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f= f[:-4]+'0.txt'
		if db=='cmupie':
			ll = pickle.load(open(f, 'rb'))[:config.splits[db_i][1]]
			used_IDs = [a[0,0] for a in ll]
			f = os.path.join(config.data_dir, 'data_info', db+'_label_list_all.txt')

			ll = pickle.load(open(f, 'rb'))
			labels = []	
			for a in ll:
				if not a[0,0] in used_IDs: labels.append(a)
		else:	
			labels = pickle.load(open(f, 'rb'))[config.splits[db_i][1]:]
		cc = 0
		for usr_label in labels:
			ID = usr_label[0][0]
			f = os.path.join(config.data_dir, db, 'usr_' + str(ID)+'.txt')
			img = pickle.load(open(f, 'rb'))
			img = np.array(img)
			if len(img.shape)<4: img = img[:, np.newaxis]
			img = img/255.0*2-1
			img = torch.Tensor(img).to(device)
			x_illu, x_pose, x_id, x_source = G.forward_enc(img)
			x_illu = x_illu.data.cpu().numpy()
			x_pose = x_pose.data.cpu().numpy()
			x_id = x_id.data.cpu().numpy()
			if config.dim_source>0: x_source = x_source.data.cpu().numpy()
	
			if len(Label)==0:
				Feat = [x_illu, x_pose, x_id, x_source] if config.dim_source>0 else [x_illu, x_pose, x_id]
				Label = usr_label
			else:
				Feat[0] = np.append(Feat[0], x_illu, 0)
				Feat[1] = np.append(Feat[1], x_pose, 0)
				Feat[2] = np.append(Feat[2], x_id, 0)
				if config.dim_source>0: Feat[3] = np.append(Feat[3], x_source, 0)
				Label = np.append(Label, usr_label, 0)
			cc += len(usr_label)
			if cc >N[db_i]: break
		print(db, cc)

	# tsne show
	# tsne map
	for i in range(len(Feat)):
		Feat[i] = TSNE(n_components=2).fit_transform(Feat[i])

	print('# images', len(Label), '#feats', len(Feat[0]))
	result= {'Feat': Feat, 'Label':Label}
	with open(os.path.join(evaluate_dir, 'tsne_data.txt'), 'wb') as fp:
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

	# global information
	'''
	import itertools
	id_markers = itertools.cycle(('+', 'x', 'd', '.', 'o', '8','s','p', '*'))
	id_colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']
	id_colors_cyc = itertools.cycle(id_colors)
	db_colors = ['#7fc97f', '#beaed4', '#fdc086']
	
	illu_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
			i''
	 config.data_dir[-1]=='2':
		illu_labels = ['-90', '-45', '0', '45', '90']
		pose_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
		pose_labels = ['-45', '-30', '-15', '0', '15', '30', '45']
	elif config.data_dir[-1]=='3':
		illu_labels = ['-67', '-30', '0', '30', '67']
		pose_colors = illu_colors
		pose_labels = ['-60', '-15', '0', '15', '60']
	session_colors = ['#1b9e77','#d95f02']
	'''
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
