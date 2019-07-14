import torch
import numpy as np
import os
import pickle
import random
import argparse
import importlib
from time import time
from sklearn.decomposition import PCA
from sklearn import svm
from models.model_map import Manipulator


def get_train_data(dbs, splits):
	train_x = []
	train_y = [] 
	for db_i, db in enumerate(dbs):
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f=f[:-4]+'0.txt'
		ll = pickle.load(open(f, 'rb'))[splits[db_i]:]
		
		for usr_l in ll:
			ID = usr_l[0, 0]
			f = os.path.join(config.data_dir, db, 'usr_'+str(ID)+'.txt')
			imgs = pickle.load(open(f, 'rb'))
			train_x += imgs
			train_y.append(usr_l)
	train_x = np.array(train_x)
	train_y = np.concatenate(train_y, 0)
	train_x = train_x.reshape(len(train_x), -1)
	return train_x, train_y

def train_pca(train_x):
	start_time = time()
	m_PCA = PCA(n_components=config.pca_dim, whiten=True, svd_solver='randomized')
	train_x_pca = m_PCA.fit_transform(train_x)
	return m_PCA, train_x_pca

def train_classifier(train_x_pca,train_y ):
	start_time = time()
	clf_illu =svm.SVC()
	clf_illu.fit(train_x_pca, train_y[:, 1])

	start_time = time()
	clf_pose = svm.SVC()
	clf_pose.fit(train_x_pca, train_y[:, 2])

	return clf_illu, clf_pose


def train(dbs, splits):
	train_x, train_y = get_train_data(dbs, splits) 
	m_PCA , train_x_pca = train_pca(train_x)
	clf_illu, clf_pose = train_classifier(train_x_pca, train_y)

	return m_PCA, clf_illu, clf_pose

def confusion_matrix(pred, gt, n):
	res = np.zeros((n, n))
	for i, j in zip(pred, gt):
		res[i, j] += 1
	return res

def accu_from_CM(m):
	n = np.sum(m[:])
	acc_n = np.sum(np.array([m[i,i] for i in range(len(m))]), 0)
	return n, acc_n

def test(m_PCA, clf_illu, clf_pose, dbs, splits):
	# probes
	results_map_illu = np.zeros((config.n_illu, config.n_illu))
	results_map_pose = np.zeros((config.n_pose, config.n_pose))

	for db_i, db in enumerate(dbs):
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f=f[:-4]+'0.txt'
		ll = pickle.load(open(f, 'rb'))[splits[db_i]:]
		
		for usr_l in ll:
			ID = usr_l[0, 0]
			f = os.path.join(config.data_dir, db, 'usr_'+str(ID)+'.txt')
			imgs = pickle.load(open(f, 'rb')) 
			imgs = np.array(imgs)

			# rec 
			imgs = imgs/255*2-1
			imgs = torch.Tensor(imgs).to(device)
			x_illu, x_pose, x_id, x_lat = G.forward_enc(imgs)
			rec_imgs = G.forward_dec(x_illu, x_pose, x_id, x_lat).add(1).div(2).mul(255).data.cpu().numpy()
			x = m_PCA.transform(rec_imgs.reshape(len(rec_imgs), -1))
			pred_illu_rec = clf_illu.predict(x)
			pred_pose_rec = clf_pose.predict(x)

			# map illu
			rot = torch.zeros(len(imgs), config.n_illu).to(device)
			for i in range(config.n_illu):
				rot[:] = 0
				rot[:, i] = 1
				x = Map_illu(x_illu, rot)
				y = G.forward_dec(x, x_pose, x_id, x_lat).add(1).div(2).mul(255).data.cpu().numpy()
				x = m_PCA.transform(y.reshape(len(y), -1))
				pred_illu = clf_illu.predict(x)
				pred_pose = clf_pose.predict(x)

				# revise rec image
				for ii in range(len(imgs)):
					if usr_l[ii, 1]==i:
						pred_illu[ii] = pred_illu_rec[ii]
						pred_pose[ii] = pred_pose_rec[ii]
	
				y = np.zeros(len(imgs), dtype=np.int16)
				y[:]=i
				res = confusion_matrix(pred_illu, y, config.n_illu)
				results_map_illu += res
				res = confusion_matrix(pred_pose, usr_l[:, 2], config.n_pose)
				results_map_pose += res
			rot = torch.zeros(len(imgs), config.n_pose).to(device)
			for i in range(config.n_pose):
				rot[:] = 0
				rot[:, i] = 1
				x = Map_pose(x_pose, rot)
				y = G.forward_dec(x_illu, x, x_id, x_lat).add(1).div(2).mul(255).data.cpu().numpy()
				x = m_PCA.transform(y.reshape(len(y), -1))
				pred_illu = clf_illu.predict(x)
				pred_pose = clf_pose.predict(x)
				
				# revise rec image
				for ii in range(len(imgs)):
					if usr_l[ii, 2]==i:
						pred_illu[ii] = pred_illu_rec[ii]
						pred_pose[ii] = pred_pose_rec[ii]
				
				res = confusion_matrix(pred_illu, usr_l[:, 1], config.n_illu)
				results_map_illu += res
				y = np.zeros(len(imgs), dtype= np.int16)
				y[:]=i
				res = confusion_matrix(pred_pose, y, config.n_pose)
				results_map_pose += res

	## results
	n2, acc2 = accu_from_CM(results_map_illu)
	print('illu classification:{}, {}, {:0.4f}'.format(acc2, n2, acc2/n2))
	n2, acc2 = accu_from_CM(results_map_pose)
	print('pose classification:{}, {}, {:0.4f}'.format(acc2, n2, acc2/n2))

def arg_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default ='net_res', help='trained model name')
	parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='data directory')
	parser.add_argument('--data_dir', type=str, default='/data/lijing/data_Oct', help='data directory')
	# do not change following
	parser.add_argument('--net_gen_name', type=str, default='model_gen_1', help='G network file')
	parser.add_argument('--gen_norm', type=str, default='instance', help='normalization method of G')
	parser.add_argument('--g_conv_dim', type=int, default=16, help='feature maps of networks')
	parser.add_argument('--dim_attr', type=int, default=64, help='dim of attr feature')
	parser.add_argument('--dim_id', type=int, default=256, help='dim of ID feat')
	parser.add_argument('--dim_source', type=int, default=64, help='dim of latent feat')
	parser.add_argument('--image_c', type=int, default=1, help='image channels')
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
	parser.add_argument('--pca_dim', type=int, default=100, help='pca components')

	args = parser.parse_args()
	args.n_illu = 5
	if 'set2' in args.networks_dir:
		args.n_pose = 7
		args.datasets = ['caspeal', 'multipie', 'cmupie']
		args.splits = [200, 200, 20]
		args.data_dir = os.path.join(args.data_dir, 'batch_set2')
	else:
		args.n_pose=5
		args.datasets =['multipie', 'caspeal']
		args.splits = [200, 500]
		args.data_dir = os.path.join(args.data_dir, 'batch_set3')

	return args 

if __name__=='__main__':
	config = arg_config()
	print('attribute classifier for synthesized images, model', config.model)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# load models 
	gen_mod = importlib.import_module('models.'+ config.net_gen_name)
	G = gen_mod.Generator(config.image_c, config.g_conv_dim, config.dim_attr, config.dim_id, config.dim_source, config.gen_norm)
	G.to(device).eval()
	G.load_state_dict(torch.load(os.path.join(config.networks_dir, config.model, 'models/G.ckpt'), map_location= lambda storage, loc:storage))

	Map_illu = Manipulator(config.n_illu, config.dim_attr)
	Map_pose = Manipulator(config.n_pose, config.dim_attr)
	Map_illu.to(device).eval()
	Map_pose.to(device).eval()
	Map_illu.load_state_dict(torch.load(os.path.join(config.networks_dir, config.model, 'models/Map_illu.ckpt'), map_location= lambda storage, loc:storage))
	Map_pose.load_state_dict(torch.load(os.path.join(config.networks_dir, config.model, 'models/Map_pose.ckpt'), map_location= lambda storage, loc:storage))
	
	'''
	print('----------------------------train on two partially-attributed datasets--------------------------')
	m_pca, clf_illu, clf_pose = train(config.datasets[:2], config.splits[:2])
	print('test on two partially-attributed datasets')
	test(m_pca, clf_illu, clf_pose, config.datasets[:2], config.splits[:2])
	'''

	print('----------------------------train all datasets--------------------------')
	m_pca, clf_illu, clf_pose = train(config.datasets, config.splits)
	print('test on two partially-attributed datasets')
	test(m_pca, clf_illu, clf_pose, config.datasets[:2], config.splits[:2])
	'''

	print('----------------------------train joint datasets--------------------------')
	m_pca, clf_illu, clf_pose = train([config.datasets[-1]], [config.splits[-1]])
	test(m_pca, clf_illu, clf_pose, config.datasets[:-1], config.splits[:-1])


	print('-----------pca full -------classifier two-------')
	x1, __ = get_train_data(config.datasets, config.splits)
	pca_all, __ = train_pca(x1)
	x2, y2 = get_train_data(config.datasets[:-1], config.splits[:-1])
	x2 = pca_all.transform(x2)
	clf_illu, clf_pose = train_classifier(x2, y2)
	test(pca_all, clf_illu, clf_pose, config.datasets[:-1], config.splits[:-1])
	'''
