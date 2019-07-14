import os
import pickle
import numpy as np
import argparse

import matplotlib as mpl 
mpl.use('Agg')
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy import interp

def NN_predict(templates, probes):
	Dist_matrix = -2* probes.dot(templates.T) + np.sum(templates**2,1) + np.tile(np.sum(probes**2, 1), (templates.shape[0], 1)).T
	result = np.argmin(Dist_matrix, 1)
	return result

def NN_distance(templates, probes):
	Dist_matrix = -2* probes.dot(templates.T) + np.sum(templates**2,1) + np.tile(np.sum(probes**2, 1), (templates.shape[0], 1)).T
	return Dist_matrix

def index_offset(db_i):
	if db_i==0:
		a, b = 2, config.n_illu+2
	elif db_i==1:
		a, b = 2+config.n_illu, config.n_illu+config.n_pose+2
	else:
		a, b = 2, config.n_illu+config.n_pose+2
	return a, b

def compare_feat_list_genuine(feats, db_i):
	# input feats [ feat_org, feat_rec, feat_map ,...]
	gen_org, gen_cross = [], []

	# org   NX(N-1)
	Dist_matrix = NN_distance(feats[0], feats[0])
	usr_n = len(feats[0])
	gen_org += [Dist_matrix[j, k] for j in range(usr_n-1) for k in range(j+1, usr_n)]

	# cross NXN X (n_illu+n_pose+1)
	a, b = index_offset(db_i)
	for j in range(a, b):
		Dist_matrix = NN_distance(feats[0],feats[j])
		gen_cross += [a for a in Dist_matrix.ravel()]

	return gen_org, gen_cross

def compare_feat_list_imposter(feats_1, feats_2, db_1, db_2):
	# input feats [ feat_org, feat_rec, feat_map ,...]
	imp_org, imp_cross = [], []

	# org  		N1xN2
	Dist_matrix = NN_distance(feats_1[0], feats_2[0])
	imp_org += [k for k in Dist_matrix.ravel()]

	# cross  	N1XN2 X (n_illu + n_pose + 1) X2
	a,b = index_offset(db_1)
	for j in range(a, b):
		Dist_matrix = NN_distance(feats_2[0], feats_1[j])
		imp_cross +=[a for a in Dist_matrix.ravel()]
	a,b = index_offset(db_2)
	for j in range(a, b):
		Dist_matrix = NN_distance(feats_1[0],feats_2[j])
		imp_cross += [a for a in Dist_matrix.ravel()]
	return imp_org,  imp_cross

def verify_faces_new():
	Feats = []
	db_info = []
	for i, db in enumerate(config.datasets):
		# load  test subjects
		f = os.path.join(config.base_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f = f[:-4]+'0.txt'
		ll = pickle.load(open(f, 'rb'))
		if db =='cmupie':
			train_ids = [a[0,0] for a in ll[:config.splits[i]]]
			f = os.path.join(config.base_dir, 'data_info', 'cmupie_label_list_all.txt')
			l = pickle.load(open(f, 'rb'))
			ll = []
			for a in l:
				if not a[0,0] in train_ids: ll.append(a)
		else:
			ll = ll[config.splits[i]:]
		IDs = [a[0,0] for a in ll]

		print('db:{}, subject:{}'.format(db, len(ll)))

		# load feats
		for ii in IDs:
			f = os.path.join(config.feat_dir, db, 'usr_'+str(ii)+'.txt')
			feat = pickle.load(open(f, 'rb'))
			Feats.append(feat)
			db_info.append(i)
	
	# compare feat
	genuine_distance_list_org, imposter_distance_list_org = [], []
	genuine_distance_list_cross, imposter_distance_list_cross = [], []
	N = len(Feats)
	for i in range(N-1):
		# genuine
		gen_org, gen_cross = compare_feat_list_genuine(Feats[i], db_info[i])
		genuine_distance_list_org += gen_org
		genuine_distance_list_cross += gen_cross

		## imposter
		for j in range(i+1, N):
			imp_org, imp_cross = compare_feat_list_imposter(Feats[i], Feats[j], db_info[i], db_info[j])
			imposter_distance_list_org += imp_org
			imposter_distance_list_cross += imp_cross

	gen_org, gen_cross = compare_feat_list_genuine(Feats[-1], db_info[-1])
	genuine_distance_list_org += gen_org
	genuine_distance_list_cross += gen_cross

	'''
	print('org roc, genuine, imposter', len(genuine_distance_list_org), len(imposter_distance_list_org))
	gen_y = np.ones(len(genuine_distance_list_org))
	imp_y = np.zeros(len(imposter_distance_list_org))
	fpr_org, tpr_org, __ = roc_curve(np.concatenate([gen_y, imp_y]), -np.array(genuine_distance_list_org+imposter_distance_list_org))
	auc_org = auc(fpr_org, tpr_org)
	print('accu', auc_org)
	'''

	print('map roc, genuine, imposter', len(genuine_distance_list_cross), len(imposter_distance_list_cross))
	gen_y = np.ones(len(genuine_distance_list_cross))
	imp_y = np.zeros(len(imposter_distance_list_cross))
	fpr_cross, tpr_cross, __ = roc_curve(np.concatenate([gen_y, imp_y]), -np.array(genuine_distance_list_cross + imposter_distance_list_cross))
	auc_cross = auc(fpr_cross, tpr_cross)
	print('accu', auc_cross)

	imp_y = np.zeros(len(imposter_distance_list_org))
	fpr_cross, tpr_cross, __ = roc_curve(np.concatenate([gen_y, imp_y]), -np.array(genuine_distance_list_cross + imposter_distance_list_org))
	auc_cross = auc(fpr_cross, tpr_cross)
	print('cross accu', auc_cross)
	'''
	n1 = len(genuine_distance_list_org)  + len(genuine_distance_list_cross)
	n2 = len(imposter_distance_list_org)+ len(imposter_distance_list_cross)
	print('all roc, genuine, imposter', n1, n2)
	gen_y = np.ones(n1)
	imp_y = np.zeros(n2)
	fpr_all, tpr_all, __ = roc_curve(np.concatenate([gen_y, imp_y]), -np.array(genuine_distance_list_cross + genuine_distance_list_org + imposter_distance_list_cross + imposter_distance_list_org))
	auc_all = auc(fpr_all, tpr_all)
	print('accu', auc_all)

	plt.figure()
	plt.plot(fpr_org, tpr_org, label ='org images (accu = {0:0.2f}'.format(auc_org), color='aqua', lw=2)

	plt.plot(fpr_cross, tpr_cross, label='generated-org (accu = {0:0.2f}'.format(auc_cross), color='deeppink', lw=2)
	plt.plot(fpr_all, tpr_all, label='all images (accu = {0:0.2f}'.format(auc_all), color='navy', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	save_info = [config.networks_dir, config.feat]
	plt.savefig(os.path.join(save_dir, '_'.join(save_info) + '.png'))
	'''
def arg_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--set', type=int, default=2, help='data setting')
	parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='model name')
	parser.add_argument('--model', type=str, default='net_v0', help='model name')
	parser.add_argument('--feat', type=str, default='lightCNN', help='feat typ, lightCNN, OpenFace, VGGFace')
	parser.add_argument('--base_dir', type=str, default='/data/lijing/data_Oct/', help='probably do not change this directory')
	parser.add_argument('--classifier', type=str, default='NN', help=' NN, SVM, KNN...')
	parser.add_argument('--gpu', type=int, default=0)
	args = parser.parse_args()
	if args.set==2:
		args.base_dir = os.path.join(args.base_dir, 'batch_set2')
		args.feat_dir = os.path.join(args.base_dir, args.networks_dir, args.model, args.feat+'_feat')
		args.datasets = ['caspeal', 'multipie', 'cmupie']
		args.splits = [200, 200, 20]
		args.n_illu = 5
		args.n_pose = 7
	elif args.set==3:
		args.base_dir = os.path.join(args.base_dir, 'batch_set3')
		args.feat_dir = os.path.join(args.base_dir, args.model, args.feat+'_feat')
		args.datasets = ['multipie', 'caspeal', 'cmupie']
		args.splits = [250, 800, 35]
		args.n_illu = 5
		args.n_pose = 5
	return args

if __name__=='__main__':
	config = arg_config()
	os.environ['CUDA_VISIBLE_DEVICES']=str(config.gpu)
	print(config)

	verify_faces_new()
