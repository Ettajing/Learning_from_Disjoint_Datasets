'''
evaluate the identity consistency between generated image and org images
input: 	batches of images (of each subject)
output:	face_teat of input images and result images
'''

import torch
import numpy as np 
import os
import pickle
import importlib 
import argparse

from code import options_set2 as options
from models.model_map import Manipulator
from evaluate.FaceFeat import Extractor

def data_preprocess(x):
	x = np.array(x)
	if len(x.shape)==3: x = x[:, np.newaxis]
	x = x/255.0 *2 -1
	return torch.Tensor(x).to(device)

def extract_feat(db, outdir_db):
	dd = os.path.join(config.data_dir, db)

	for f in os.listdir(dd):
		x = pickle.load(open(os.path.join(dd, f), 'rb'))
		x = data_preprocess(x)

		feat = FeatExtractor.extract_feat(x).detach().cpu().numpy()

		result = [feat]
		x_illu, x_pose, x_id, x_source = G.forward_enc(x)
		n = len(x)
		feat = FeatExtractor.extract_feat(G.forward_dec(x_illu, x_pose, x_id, x_source))
		result.append(feat.detach().cpu().numpy())

		# manipulate illu
		for i in range(config.n_illu):
			rot = torch.zeros(n, config.n_illu)
			rot[:, i]=1
			x = Map_illu(x_illu, rot.to(device))
			feat = FeatExtractor.extract_feat(G.forward_dec(x, x_pose, x_id, x_source))
			result.append(feat.detach().cpu().numpy())

		for i in range(config.n_pose):
			rot = torch.zeros(n, config.n_pose)
			rot[:, i]=1
			x = Map_pose(x_pose, rot.to(device))
			feat = FeatExtractor.extract_feat(G.forward_dec(x_illu, x, x_id, x_source))
			result.append(feat.detach().cpu().numpy())

		with open(os.path.join(outdir_db, f), 'wb') as fp:
			pickle.dump(result, fp)
	
def arg_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--set', type=int, default=2, help='data setting')
	parser.add_argument('--model', type=str, default ='net_v0', help='trained model name')
	parser.add_argument('--feat', type=str, default= 'lightCNN', help ='feat_type, lightCNN, VGGFace, OpenFace')		
	parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='data directory')
	parser.add_argument('--data_dir', type=str, default='/data/lijing/data_Oct/', help='data directory')
	# do not change following
	parser.add_argument('--net_gen_name', type=str, default='model_gen_0', help='G network file')
	parser.add_argument('--gen_norm', type=str, default='instance', help='normalization method of G')
	parser.add_argument('--d_conv_dim', type=int, default=16, help='feature maps of networks')
	parser.add_argument('--g_conv_dim', type=int, default=16, help='feature maps of networks')
	parser.add_argument('--dim_attr', type=int, default=64, help='dim of attr feature')
	parser.add_argument('--dim_id', type=int, default=256, help='dim of ID feat')
	parser.add_argument('--dim_source', type=int, default=64, help='dim of latent feat')
	parser.add_argument('--image_c', type=int, default=1, help='image channels')
	parser.add_argument('--gpu', type=int, default=0, help='cuda')
	
	args = parser.parse_args()
	if args.set==2:
		args.n_illu = 5
		args.n_pose = 7
		args.datasets = ['caspeal', 'multipie', 'cmupie']	
	else:
		args.n_illu = 5
		args.n_pose = 5
		args.datasets = ['multipie', 'caspeal', 'cmupie']
	args.data_dir = os.path.join(args.data_dir, 'batch_set'+str(args.set))

	return args

if __name__=='__main__':
	config = arg_config()

	print('extract', config.feat,' feat for model ', config.model)
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
	#
	FeatExtractor = Extractor(config.feat, device)
	#
	#outdir = os.path.join(config.data_dir, config.model, config.feat+'_feat')
	outdir = os.path.join(config.data_dir, config.networks_dir, config.model, config.feat+'_feat')
	if not os.path.exists(outdir): os.makedirs(outdir)

	for db in config.datasets:
		outdir_db = os.path.join(outdir, db)
		if not os.path.exists(outdir_db): os.makedirs(outdir_db)

		extract_feat(db, outdir_db)
