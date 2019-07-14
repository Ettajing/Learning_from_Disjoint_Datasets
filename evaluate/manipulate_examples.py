from torchvision.utils import save_image
import torch
import numpy as np
import os
import pickle
import random
import importlib
import argparse
from models.model_map import Manipulator
from PIL import Image

def reconstruct_img(G, img):
	#img = img/255.0*2-1
	img = torch.Tensor(img).to(device)

	x_fake = G(img)
	result = [img, x_fake]
	result = torch.cat(result, dim=3).data.cpu()
	#denorm
	result = (result + 1) / 2
	result = result.clamp_(0, 1)
	return result
def synthesize_img(img):
	img = torch.Tensor(img).to(device)
	x_illu, x_pose, x_id, x_source =G.forward_enc(img)
	x_rec = G.forward_dec(x_illu, x_pose, x_id, x_source)
	result = [img, x_rec]

	for k in range(config.n_illu):
		rot = torch.zeros(len(img), config.n_illu)
		rot[:, k]=1
		rot = rot.to(device)
		fake_illu = Map_illu(x_illu, rot)
		x_fake = G.forward_dec(fake_illu, x_pose, x_id, x_source)	
		result.append(x_fake)

	for k in range(config.n_pose):
		rot = torch.zeros(len(img), config.n_pose)
		rot[:, k]=1
		rot = rot.to(device)
		fake_pose = Map_pose(x_pose, rot)
		x_fake = G.forward_dec(x_illu, fake_pose, x_id, x_source)
		result.append(x_fake)
	
	result = torch.cat(result, dim=3).data.cpu()
	#denorm
	result = (result + 1) / 2
	result = result.clamp_(0, 1)
	return result

def arg_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='data setting')
	parser.add_argument('--model', type=str, default ='net_res', help='trained model name')
	parser.add_argument('--data_dir', type=str, default='/data/lijing/data_Oct/batch_set2', help='data directory')
	# do not change following
	parser.add_argument('--net_gen_name', type=str, default='model_gen_1', help='G network file')
	parser.add_argument('--gen_norm', type=str, default='instance', help='normalization method of G')
	parser.add_argument('--g_conv_dim', type=int, default=16, help='feature maps of networks')
	parser.add_argument('--dim_attr', type=int, default=64, help='dim of attr feature')
	parser.add_argument('--dim_id', type=int, default=256, help='dim of ID feat')
	parser.add_argument('--dim_source', type=int, default=64, help='dim of latent feat')
	parser.add_argument('--image_c', type=int, default=1, help='image channels')
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use')

	args = parser.parse_args()
	args.n_illu = 5
	args.n_pose = 7
	args.datasets = ['caspeal','multipie','cmupie']
	args.splits = [200, 200, 20]
	return args 

def load_faces():
	# detect indice
	img_list = []

	# caspeal
	f = os.path.join(config.data_dir, 'data_info','caspeal_label_list.txt')
	if not os.path.exists(f): f= f[:-4]+'0.txt'
	ll = pickle.load(open(f, 'rb'))[config.splits[0]:]
	dd = os.path.join(config.data_dir, 'caspeal')
	for a in ll:
		f = os.path.join(dd, 'usr_'+str(a[0,0]) +'.txt')
		imgs = pickle.load(open(f, 'rb'))
		img_list = img_list + imgs
	# multipie
	f = os.path.join(config.data_dir, 'data_info','multipie_label_list.txt')
	if not os.path.exists(f): f= f[:-4]+'0.txt'
	ll = pickle.load(open(f, 'rb'))[config.splits[1]:]
	dd = os.path.join(config.data_dir, 'multipie')
	for a in ll:
		f = os.path.join(dd, 'usr_'+str(a[0,0]) +'.txt')
		imgs = pickle.load(open(f, 'rb'))
		img_list += imgs[1:4]
	# cmupie		
	f = os.path.join(config.data_dir, 'data_info', 'cmupie_label_list.txt')
	if not os.path.exists(f): f= f[:-4]+'0.txt'
	ll = pickle.load(open(f, 'rb'))[:config.splits[2]]
	train_ids = [a[0,0] for a in ll]
	
	f = os.path.join(config.data_dir, 'data_info', 'cmupie_label_list_all.txt')
	ll = pickle.load(open(f, 'rb'))	
	IDs = []
	for a in ll:
		if not a[0,0] in train_ids:IDs.append(a[0,0])
	
	dd = os.path.join(config.data_dir, 'cmupie')
	for usr_id in IDs:
		f = os.path.join(dd, 'usr_'+str(usr_id)+'.txt')
		imgs = pickle.load(open(f, 'rb'))
		n = len(imgs)//2
		img_list += imgs[:n]

	img_list = np.array(img_list)
	print(img_list.shape)
	if len(img_list.shape)==3: img_list = img_list[:,np.newaxis]
	return img_list
	
if __name__=='__main__':
	config = arg_config()
	print(config.networks_dir, config.model)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	outdir = os.path.join(config.networks_dir, config.model, 'evaluate')
	if not os.path.exists(outdir): os.makedirs(outdir)

	#load models
	gen_mod = importlib.import_module('models.'+config.net_gen_name)
	model_path = os.path.join(config.networks_dir, config.model, 'models')
	# build model
	imgsize = 128
	G = gen_mod.Generator(config.image_c, config.g_conv_dim, config.dim_attr, config.dim_id, config.dim_source, config.gen_norm)
	G.to(device).eval()
	G.load_state_dict(torch.load(os.path.join(model_path, 'G.ckpt'), map_location=lambda storage, loc:storage))
	
	Map_illu = Manipulator(config.n_illu, config.dim_attr)
	Map_pose = Manipulator(config.n_pose, config.dim_attr)
	Map_illu.to(device)#.eval()
	Map_pose.to(device)#.eval()
	Map_illu.load_state_dict(torch.load(os.path.join(model_path, 'Map_illu.ckpt'), map_location=lambda storage, loc:storage))
	Map_pose.load_state_dict(torch.load(os.path.join(model_path, 'Map_pose.ckpt'), map_location=lambda storage, loc:storage))


	images = load_faces()
	print(len(images))
	images = images/255.0*2-1
	images = torch.Tensor(images)

	i = 0
	while len(images)>20:
		batch = images[:20]
		images = images[20:]
		batch = synthesize_img(batch)
		save_image(batch, os.path.join(outdir, str(i)+'.png'), nrow=1, padding=0)
		i += 1

	if len(images)>0: 
		result = synthesize_img(images)
		save_image(result, os.path.join(outdir, str(i)+'.png'), nrow=1, padding=0)

	'''
	example_dd = 'net_set2/examples/org_img'
	out_dd = 'net_set2/examples/syn_img'
	for db_name in os.listdir(example_dd): 
		img_list = []
		dd = os.path.join(example_dd, db_name)
		for  f in os.listdir(dd):
			img = Image.open(os.path.join(dd, f)).convert('L')
			img = np.array(img, dtype=np.float16)
			img_list.append(img[np.newaxis,:])
	
		img_list = np.array(img_list)
		x = img_list/255.0*2-1
		x = torch.Tensor(x)

		#===================================
		result = synthesize_img(x)
		save_image(result, os.path.join(out_dd, config.name+'_'+db_name+'.jpg'), nrow=1, padding=0)
	'''

