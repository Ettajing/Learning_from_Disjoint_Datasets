import os
import pickle
import argparse
import importlib

import torch
import numpy as np 
from scipy import linalg
from torch.autograd import Variable
import torch.nn.functional as F

from evaluate.inception import InceptionV3
from models.model_map import Manipulator

def compute_activation_statistics_org(db, save_f, split, tag=False):
	'''
	compute statistics for each dataset 
	'''
	print('compute activation statistics for original images')

	results = []
	
	dd = os.path.join(config.data_dir, db)
	f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
	if not os.path.exists(f): f= f[:-4]+'0.txt'
	ll = pickle.load(open(f, 'rb'))
	used_id = [a[0,0] for a in ll[split:]]

	if tag: dd = '/data/lijing/data_Oct/batch_full_multipie/multipie'

	if db =='cmupie':
		train_ids = [a[0,0] for a in ll[:split]]
		f = os.path.join(config.data_dir, 'data_info', 'cmupie_label_list_all.txt')
		used_id = []
		ll = pickle.load(open(f, 'rb'))
		for l in ll:
			if not l[0,0] in train_ids: used_id.append(l[0,0]) 
	print(db, '# ids', len(used_id))

	for usr_i in used_id:
		x = pickle.load(open(os.path.join(dd, 'usr_'+str(usr_i) +'.txt'), 'rb'))
		x = np.array(x)
		x /= 255
		if len(x.shape)==3: x= x[:, np.newaxis]
		x = torch.Tensor(x).to(device)
		pred = incept_model(x.repeat(1, 3, 1, 1))[0]
		if pred.shape[2] !=1 or pred.shape[3]!=1:
			pred = F.adaptive_avg_poo2d(pred, output_size=(1, 1))
		results.append(pred.cpu().data.numpy().reshape(len(x), -1))

	# pred
	res2 = np.concatenate(results)
	m1 = np.mean(res2, axis=0)
	sigma1 = np.cov(res2, rowvar=False)
	output={'mean':m1, 'sigma':sigma1}

	# save
	with open(save_f, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compute_activation_statistics_map(db, save_f, split, attr):
	'''
	compute statistics for result images
	'''
	print('compute activation statistics for synthesized images')
	results_illu_map = [] # rec, illu_map, pose_map
	results_pose_map = []

	dd = os.path.join(config.data_dir, db)
	f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
	if not os.path.exists(f): f= f[:-4]+'0.txt'
	ll = pickle.load(open(f, 'rb'))

	if db =='cmupie':
		used_ids = [a[0,0] for a in ll[:split]]
		f = os.path.join(config.data_dir, 'data_info', 'cmupie_label_list_all.txt')
		labels = pickle.load(open(f, 'rb'))
		ll =[]
		for l in labels:
			if not l[0,0] in used_ids: ll.append(l)	
	else:
		ll = ll[split:]


	for usr_l in ll:
		usr_i = usr_l[0,0]
		x = pickle.load(open(os.path.join(dd, 'usr_'+str(usr_i) +'.txt'), 'rb'))
		x = np.array(x)
		x /= 255*2 - 1
		if len(x.shape)==3: x= x[:, np.newaxis]
		x = torch.Tensor(x).to(device)
		n = len(x)

		## manipulate
		x_illu, x_pose, x_id, x_lat = G.forward_enc(x)
		x_rec = G.forward_dec(x_illu, x_pose, x_id, x_lat)
		pred = incept_model(x_rec.add(1).div(2).repeat(1, 3, 1, 1))[0]
		if pred.shape[2] !=1 or pred.shape[3]!=1:
			pred = F.adaptive_avg_poo2d(pred, output_size=(1, 1))
		pred_rec = pred.data.cpu().numpy().reshape(n ,-1)
		#results.append(pred.cpu().data.numpy().reshape(n, -1))

		rot = torch.zeros(n, config.n_illu).to(device)
		for j in range(config.n_illu):
			rot[:] = 0
			rot[:, j] = 1
			x = Map_illu(x_illu, rot)
			y = G.forward_dec(x, x_pose, x_id, x_lat).add(1).div(2).repeat(1, 3, 1, 1)
			pred = incept_model(y)[0]
			if pred.shape[2]!= 1 or pred.shape[3]!=1:
				pred = F.adaptive_avg_poo2d(pred, output_size=(1, 1))
			
			pred = pred.data.cpu().numpy().reshape(n, -1)
			# revise
			for ii in range(n):
				if usr_l[ii, 1] == j: pred[ii] = pred_rec[ii] 
			results_illu_map.append(pred)

		rot = torch.zeros(n, config.n_pose).to(device)
		for j in range(config.n_pose):
			rot[:]= 0 
			rot[:, j] =1 
			x = Map_pose(x_pose, rot)
			y = G.forward_dec(x_illu, x, x_id, x_lat).add(1).div(2).repeat(1, 3, 1, 1)
			pred  = incept_model(y)[0]
			if pred.shape[2]!=1 or pred.shape[3]!=1:
				pred = F.adaptive_avg_poo2d(pred, output_size=(1,1))

			pred = pred.data.cpu().numpy().reshape(n, -1)
			# revise
			for ii in range(n):
				if usr_l[ii, 2] == j: pred[ii] = pred_rec[ii] 
			results_pose_map.append(pred)

	output = {}
	output['illu'] = {}
	res = np.concatenate(results_illu_map)
	m1 = np.mean(res, axis=0)
	sigma1 = np.cov(res, rowvar=False)
	output['illu'] = {'mean': m1, 'sigma': sigma1}

	output['pose'] = {}
	res = np.concatenate(results_pose_map)
	m1 = np.mean(res, axis=0)
	sigma1 = np.cov(res, rowvar=False)
	output['pose'] = {'mean': m1, 'sigma': sigma1}

	output['all'] = {}
	res = np.concatenate(results_illu_map + results_pose_map)
	m1 = np.mean(res, axis=0)
	sigma1 = np.cov(res, rowvar=False)
	output['all'] = {'mean': m1, 'sigma': sigma1}

	# save
	with open( save_f, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def compute_fid():
	# paritally-attributed multipie 
	f = os.path.join(save_dir, 'partial_multipie_org.pickle')
	if not os.path.exists(f): compute_activation_statistics_org('multipie', f, config.splits[1])
	data_org = pickle.load(open(f, 'rb'))
	
	f = os.path.join(save_dir, config.model + '_multipie_map.pickle')		
	if not os.path.exists(f): compute_activation_statistics_map('multipie', f, config.splits[1], 'pose')
	data_map = pickle.load(open(f, 'rb'))['pose']
	
	fid_value = calculate_frechet_distance(data_org['mean'], data_org['sigma'], data_map['mean'], data_map['sigma'])
	print('fid between parital multipie org and map:{}'.format(fid_value))

	# paritally-attributed multipie 
	f = os.path.join(save_dir, 'partial_caspeal_org.pickle')
	if not os.path.exists(f): compute_activation_statistics_org('caspeal', f, config.splits[0])
	data_org = pickle.load(open(f, 'rb'))

	f = os.path.join(save_dir, config.model + '_caspeal_map.pickle')		
	if not os.path.exists(f): compute_activation_statistics_map('multipie', f, config.splits[0], 'illu')
	data_map = pickle.load(open(f, 'rb'))['illu']
	
	fid_value = calculate_frechet_distance(data_org['mean'], data_org['sigma'], data_map['mean'], data_map['sigma'])
	print('fid between parital caspeal org and map:{}'.format(fid_value))

	# fully-attributed cmupie 
	f = os.path.join(save_dir, 'full_cmupie_org.pickle')
	if not os.path.exists(f): compute_activation_statistics_org('cmupie', f, config.splits[2])
	data_org = pickle.load(open(f, 'rb'))
	
	f = os.path.join(save_dir, config.model + '_cmupie_map.pickle')		
	if not os.path.exists(f): compute_activation_statistics_map('cmupie', f, config.splits[2], '')
	data_map = pickle.load(open(f, 'rb'))['all']
	
	fid_value = calculate_frechet_distance(data_org['mean'], data_org['sigma'], data_map['mean'], data_map['sigma'])
	print('fid between cmupie org and map:{}'.format(fid_value))

	# fully-attributed multipie 
	f = os.path.join(save_dir, 'full_multipie_org.pickle')
	if not os.path.exists(f): compute_activation_statistics_org('multipie', f, config.splits[1], True)
	data_org = pickle.load(open(f, 'rb'))

	f = os.path.join(save_dir, config.model + '_multipie_map.pickle')		
	if not os.path.exists(f): compute_activation_statistics_map('multipie', f, config.splits[1], 'pose')
	data_map = pickle.load(open(f, 'rb'))['all']
	
	fid_value = calculate_frechet_distance(data_org['mean'], data_org['sigma'], data_map['mean'], data_map['sigma'])
	print('fid between full multipie org and map:{}'.format(fid_value))

def arg_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--set', type=int, default=2, help='data setting')
	parser.add_argument('--model', type=str, default ='net_v0', help='trained model name')
	parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='data directory')
	parser.add_argument('--data_dir', type=str, default='/data/lijing/data_Oct/', help='data directory')
	# do not change following
	parser.add_argument('--net_gen_name', type=str, default='model_gen_1', help='G network file')
	parser.add_argument('--gen_norm', type=str, default='instance', help='normalization method of G')
	parser.add_argument('--g_conv_dim', type=int, default=16, help='feature maps of networks')
	parser.add_argument('--dim_attr', type=int, default=64, help='dim of attr feature')
	parser.add_argument('--dim_id', type=int, default=256, help='dim of ID feat')
	parser.add_argument('--dim_source', type=int, default=64, help='dim of latent feat')
	parser.add_argument('--image_c', type=int, default=1, help='image channels')
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use')

	parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM), help='Dimensionality of Inception features to use.' 
						'By default, uses pool3 features')


	args = parser.parse_args()

	args.n_illu = 5
	if args.set==2:
		args.n_pose = 7
		args.datasets = ['caspeal', 'multipie', 'cmupie']
		args.splits = [200, 200, 20]
	else:
		args.n_pose=5
		args.datasets =['multipie', 'caspeal', 'cmupie']
		args.splits = [250, 800, 35]
	args.data_dir = os.path.join(args.data_dir, 'batch_set' + str(args.set))

	return args 

if __name__=='__main__':
	config = arg_config()
	print('compute fid scores for model ', config.model)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	save_dir = 'evaluate_multi_db_new/fids'
	if not os.path.exists(save_dir): os.makedirs(save_dir)

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

	# load incept net
	block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[config.dims]
	incept_model = InceptionV3([block_idx]).to(device).eval()

	#-------------------
	compute_fid()
