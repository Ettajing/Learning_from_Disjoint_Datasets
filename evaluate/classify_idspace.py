import torch
import numpy as np
import os
import pickle
import random
from code import options_set2_1 as options
import importlib

def Split_DB(datasets, splits, dd_1):
	'''
		for all each user in valid set, randomly select half samples as galleries, the other half as probes
	'''
		
	# partially-attributed db

	for i, db in enumerate(datasets):
		gallery_indices = []
		probe_indices = []
			
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f = f[:-4]+'0'+'.txt'
		labels = pickle.load(open(f, 'rb'))[splits[i][1]:]
		
		for usr_label in labels:
			n = len(usr_label)
			indices = np.random.permutation(n)
			n = n//2
			gallery_indices.append(indices[n:])
			probe_indices.append(indices[:n])
		with open(os.path.join(config.data_dir, 'data_info', db+'_gallery_split.txt'), 'wb') as fp:
			pickle.dump(gallery_indices, fp)
		with open(os.path.join(config.data_dir, 'data_info', db +'_probe_split.txt'), 'wb') as fp:
			pickle.dump(probe_indices, fp)
def Compare_feat(templates, probes):
	## compare 
	Dist_matrix = -2* probes.dot(templates.T) + np.sum(templates**2,1) + np.tile(np.sum(probes**2, 1), (templates.shape[0], 1)).T
	result = np.argmin(Dist_matrix, 1)
	return result


def label2onehot(n_class, y):
	t = np.array(y)
	n = len(y)
	out = torch.zeros(n, n_class)
	out[np.arange(n), t] =1
	return out 

def Classify_ID(datasets, splits):
	id_offset = [0]

	labels = []
	# train with gallery images
	n_feat = 3 if config.dim_source==0 else 4
	feat_id = [[] for i in range(n_feat)]
	label_id = []
	subject_db = []
	samples_db = []
	for i, db in enumerate(datasets):
		gallery_indices = pickle.load(open(os.path.join(config.data_dir, 'data_info', db+'_gallery_split.txt'), 'rb'))
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f=f[:-4]+'0.txt'
		db_labels = pickle.load(open(f, 'rb'))[splits[i][1]:]

		if not len(db_labels)==len(gallery_indices):
			print('error splits in ', db, len(db_labels), len(gallery_indices))

		labels.append(db_labels)
		max_id = 0
		num_subject = 0
		num_samples = 0
		for usr_ind, usr_label in zip(gallery_indices, db_labels):
			if len(usr_ind)<1: continue
			num_subject += 1
			num_samples += len(usr_ind)
			ID = usr_label[0,0]
			if ID>max_id: max_id = ID
			f = os.path.join(config.data_dir, db, 'usr_'+str(ID)+'.txt')
			img = np.array(pickle.load(open(f, 'rb')))[usr_ind]
			if len(img.shape)<4: img = img[:, np.newaxis]
			img = img/255.0*2 - 1
			img = torch.Tensor(img).to(device)

			feats = G.forward_enc(img)
			for ii in range(n_feat): feat_id[ii].append(np.mean(feats[ii].data.cpu().numpy(), 0))
			label_id.append(ID+id_offset[i])
		id_offset.append(id_offset[i] + max_id+1)
		subject_db.append(num_subject)
		samples_db.append(num_samples)
	for ii in range(n_feat): feat_id[ii] = np.array(feat_id[ii])
	label_id = np.array(label_id)
	print('total gallery subjects', len(feat_id[0]))
	for i in range(len(datasets)):
		print(datasets[i],'subject', subject_db[i],'images', samples_db[i])
	# test with probe images
	groundtruth_label = []
	pred_label = [[] for i in range(n_feat)]

	subject_db =[]
	samples_db =[]
	for i, db in enumerate(datasets):
		probe_indices = pickle.load(open(os.path.join(config.data_dir, 'data_info', db+'_probe_split.txt'), 'rb'))
		if not len(probe_indices) == len(labels[i]):
			print('error probe splits in ', db, len(probe_indices), len(labels[i]))
		
		num_subject = 0
		num_samples = 0
		for usr_ind, usr_label in zip(probe_indices, labels[i]):
			if len(usr_ind)<1:continue
			num_subject +=1
			num_samples += len(usr_ind)
			ID = usr_label[0,0]
			f = os.path.join(config.data_dir, db, 'usr_'+str(ID)+'.txt')
			img = np.array(pickle.load(open(f, 'rb')))[usr_ind]
			if len(img.shape)<4: img = img[:, np.newaxis]
			img = img/255.0*2 - 1
			img = torch.Tensor(img).to(device)

			feats = G.forward_enc(img)
			for ii in range(n_feat):
				v = feats[ii].data.cpu().numpy()
				result = Compare_feat(feat_id[ii], v)
				pred_label[ii].append(label_id[result])

			y = np.zeros(len(usr_ind))
			y[:] = ID + id_offset[i]
			groundtruth_label.append(y)

		subject_db.append(num_subject)
		samples_db.append(num_samples)
	for ii in range(n_feat): pred_label[ii] = np.concatenate(pred_label[ii], 0)
	groundtruth_label = np.concatenate(groundtruth_label, 0)

	real_c = [[0.0 for i in range(len(datasets))] for ii in range(n_feat)] 
	start = 0
	for i in range(len(datasets)):
		for j in range(start, start+samples_db[i]):
			for ii in range(n_feat):
				if pred_label[ii][j]== groundtruth_label[j]: real_c[ii][i] +=1
	
	print('probes')
	for i in range(len(datasets)):
		print(datasets[i], 'subject', subject_db[i], 'probes', samples_db[i])
		for ii in range(n_feat):
			print('feat space', ii, real_c[ii][i]/samples_db[i])

	print('all', 'subjects', np.sum(np.array(subject_db)))
	for i in range(n_feat):
		all_pred = np.sum(np.array(real_c[i]))
		print('feat space', i,all_pred/len(pred_label[i]))
	

if __name__=='__main__':
	parser = options.TrainOptions()
	config = parser.parse()
	print(config.name)	
	net_mode = importlib.import_module('models.'+config.net_gen_name)

	#Split_DB(config.datasets[:-1], config.splits[:-1])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	imgsize = 128
	G = net_mode.Generator(config.image_c, config.g_conv_dim, config.dim_attr, config.dim_id, config.dim_source, config.gen_norm)
	G.to(device).eval()
	G.load_state_dict(torch.load(os.path.join(config.networks_dir, config.name, 'models/G.ckpt'), map_location= lambda storage, loc:storage))

	if config.evaluate_map_illu<config.n_illu or config.evaluate_map_pose<config.n_pose or config.evaluate_map:
		from models.model_map import Manipulator
		Map_illu = Manipulator(config.n_illu, config.dim_attr).to(device)
		Map_pose = Manipulator(config.n_pose, config.dim_attr).to(device)
		Map_illu.load_state_dict(torch.load(os.path.join(config.networks_dir, config.name, 'models/Map_illu.ckpt'), map_location= lambda storage, loc:storage))
		Map_pose.load_state_dict(torch.load(os.path.join(config.networks_dir, config.name, 'models/Map_pose.ckpt'), map_location= lambda storage, loc:storage))
		
	Classify_ID(config.datasets[:-1], config.splits[:-1])
