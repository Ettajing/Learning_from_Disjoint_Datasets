import torch
import numpy as np
import os
import pickle
import random
from code import options_set2_1 as options
import importlib 

def Compare_feat(templates, probes):
	## compare 
	Dist_matrix = -2* probes.dot(templates.T) + np.sum(templates**2,1) + np.tile(np.sum(probes**2, 1), (templates.shape[0], 1)).T
	result = np.argmin(Dist_matrix, 1)
	return result

def Classify_attr():
	# train db
	train_db = config.datasets[-1]
	f = os.path.join(config.data_dir, 'data_info', train_db +'_label_list.txt')
	if not os.path.exists(f): f=f[:-4]+'0.txt'
	labels = pickle.load(open(f, 'rb'))[:config.splits[-1][1]]
	used_IDs = [l[0,0] for l in labels]

	f = os.path.join(config.data_dir, 'data_info', train_db +'_label_list_all.txt')
	ll = pickle.load(open(f, 'rb'))
	labels = []
	for l in ll:
		if not l[0,0] in used_IDs:
			labels.append(l)

	print('number of training subjects', len(labels))
	n_feat =3 if config.dim_source==0 else 4
	n_dims = [config.dim_attr, config.dim_attr, config.dim_id, config.dim_source]
	template_illu = [[ np.zeros(l) for i in range(config.n_illu)] for l in n_dims]
	template_pose = [[ np.zeros(l) for i in range(config.n_pose)] for l in n_dims]
	sample_illu = [0 for i in range(config.n_illu)]
	sample_pose = [0 for i in range(config.n_pose)]

	for usr_label in labels:
		ID = usr_label[0,0]	
		f = os.path.join(config.data_dir, train_db, 'usr_'+str(ID)+'.txt')
		img = np.array(pickle.load(open(f, 'rb')))
		if len(img.shape)<4: img = img[:, np.newaxis]
		img = img/255.0*2 - 1
		img = torch.Tensor(img).to(device)

		if config.evaluate_rec:
			features = G.forward_enc(G.forward(img))
		else:
			features = G.forward_enc(img)
		feats = [features[i].data.cpu().numpy() for i in range(n_feat)]

		for v in range(config.n_illu):
			ind = np.where(usr_label[:,1]==v)[0]
			if len(ind)>0:
				sample_illu[v] += len(ind)
				for j in range(n_feat):
					template_illu[j][v] += np.sum(feats[j][ind],0)
		for v in range(config.n_pose):
			ind = np.where(usr_label[:,2]==v)[0]
			if len(ind)>0:
				sample_pose[v] += len(ind)
				for j in range(n_feat):
					template_pose[j][v] += np.sum(feats[j][ind], 0)
						
	sample_illu = np.array(sample_illu)
	sample_pose = np.array(sample_pose)
	print('# illu', sample_illu)
	print('# pose', sample_pose)
	template_illu = [np.array(template_illu[i]) for i in range(n_feat)]
	for j in range(n_feat):
		div_comp = np.tile(sample_illu, (n_dims[j], 1)).T
		template_illu[j] /= div_comp
	
	template_pose =[np.array(template_pose[i]) for i in range(n_feat)] 
	for j in range(n_feat):
		div_comp = np.tile(sample_pose, (n_dims[j], 1)).T
		template_pose[j]/= div_comp	
	print('galleries')
	print(config.datasets[-1], 'illu', sample_illu, 'pose', sample_pose)

	# test with probe images
	print('testing with probes')
	result_illu, result_pose = [], []
	for i, db in enumerate(config.datasets[:-1]):
		illu_con_matrices = [np.zeros((config.n_illu, config.n_illu)) for j in range(n_feat)]
		pose_con_matrices = [np.zeros((config.n_pose, config.n_pose)) for j in range(n_feat)]
		f = os.path.join(config.data_dir, 'data_info', db+'_label_list.txt')
		if not os.path.exists(f): f = f[:-4]+'0.txt'
		labels = pickle.load(open(f, 'rb'))[config.splits[i][1]:]
		print('test db, ', db, '# subjects', len(labels)) 
		for usr_label in labels:
			ID = usr_label[0,0]	
			f = os.path.join(config.data_dir, db, 'usr_'+str(ID)+'.txt')
			img = np.array(pickle.load(open(f, 'rb')))
			if len(img.shape)<4: img = img[:, np.newaxis]
			img = img/255.0*2 - 1
			img = torch.Tensor(img).to(device)
			
			if config.evaluate_rec:
				feats = G.forward_enc(G.forward(img))
			else:
				feats = G.forward_enc(img)
			
			for j in range(n_feat):
				v = feats[j].data.cpu().numpy()
				pred_illu = Compare_feat(template_illu[j], v)
				pred_pose = Compare_feat(template_pose[j], v)
				
				for k in range(len(usr_label)):
					illu_con_matrices[j][usr_label[k, 1], pred_illu[k]] += 1
					pose_con_matrices[j][usr_label[k, 2], pred_pose[k]] += 1


		a = []
		b=[]
		for k in range(n_feat):
			#print('feat space', k)
			accu_illu = 0.0
			for j in range(config.n_illu): accu_illu += illu_con_matrices[k][j, j]
			accu_pose =0.0
			for j in range(config.n_pose): accu_pose += pose_con_matrices[k][j, j]
			n1 = np.sum(illu_con_matrices[k])
			#print(db, 'illu probes', np.sum(illu_con_matrices[k], 1), 'pred_true', accu_illu, 'accuracy rate', accu_illu/n1)
			n2 = np.sum(pose_con_matrices[k])
			#print(db, 'pose probes', np.sum(pose_con_matrices[k], 1), 'pred_true', accu_pose, 'accuracy rate', accu_pose/n2)
			a.append([accu_illu, n1])
			b.append([accu_pose, n2])	
		result_illu.append(a)
		result_pose.append(b) 

	# result of two db
	print('total result')
	for i in range(n_feat):
		accu_illu = result_illu[0][i][0] + result_illu[1][i][0]
		n1 =  result_illu[0][i][1] + result_illu[1][i][1]
		accu_pose = result_pose[0][i][0] + result_pose[1][i][0]
		n2 =  result_pose[0][i][1] + result_pose[1][i][1]
		print('feat space', i)
		print('illu, pred_true', accu_illu, 'accuracy rate', accu_illu/n1)
		print('pose, pred_true', accu_pose, 'accuracy rate', accu_pose/n2)
							
def Classify_attr_multipie():
	# train db
	f = os.path.join(config.data_dir, 'data_info', 'multipie_label_list.txt')
	if not os.path.exists(f): f = f[:-4]+'0.txt'
	labels = pickle.load(open(f, 'rb'))[config.splits[1][1]:]
	test_subjects = [l[0, 0] for l in labels]
	dd = '/data/lijing/data_Oct/batch_full_multipie/multipie'
	f = '/data/lijing/data_Oct/batch_full_multipie/data_info/multipie_label_list0.txt'
	l = pickle.load(open(f, 'rb'))
	labels = []
	for a in l:
		if a[0,0] in test_subjects: labels.append(a)
	
	train_labels = labels[:50]
	probe_labels = labels[50:]
	print('number of training subjects', len(train_labels), len(probe_labels))
	n_feat =3 if config.dim_source==0 else 4
	n_dims = [config.dim_attr, config.dim_attr, config.dim_id, config.dim_source]
	template_illu = [[ np.zeros(l) for i in range(config.n_illu)] for l in n_dims]
	template_pose = [[ np.zeros(l) for i in range(config.n_pose)] for l in n_dims]
	sample_illu = [0 for i in range(config.n_illu)]
	sample_pose = [0 for i in range(config.n_pose)]

	for usr_label in train_labels:
		ID = usr_label[0,0]	
		f = os.path.join(dd , 'usr_'+str(ID)+'.txt')
		img = np.array(pickle.load(open(f, 'rb')))
		if len(img.shape)<4: img = img[:, np.newaxis]
		img = img/255.0*2 - 1
		img = torch.Tensor(img).to(device)

		if config.evaluate_rec:
			features = G.forward_enc(G.forward(img))
		else:
			features = G.forward_enc(img)
		feats = [features[i].data.cpu().numpy() for i in range(n_feat)]

		for v in range(config.n_illu):
			ind = np.where(usr_label[:,1]==v)[0]
			if len(ind)>0:
				sample_illu[v] += len(ind)
				for j in range(n_feat):
					template_illu[j][v] += np.sum(feats[j][ind],0)
		for v in range(config.n_pose):
			ind = np.where(usr_label[:,2]==v)[0]
			if len(ind)>0:
				sample_pose[v] += len(ind)
				for j in range(n_feat):
					template_pose[j][v] += np.sum(feats[j][ind], 0)
						
	sample_illu = np.array(sample_illu)
	sample_pose = np.array(sample_pose)
	print('# illu', sample_illu)
	print('# pose', sample_pose)
	template_illu = [np.array(template_illu[i]) for i in range(n_feat)]
	for j in range(n_feat):
		div_comp = np.tile(sample_illu, (n_dims[j], 1)).T
		template_illu[j] /= div_comp
	
	template_pose =[np.array(template_pose[i]) for i in range(n_feat)] 
	for j in range(n_feat):
		div_comp = np.tile(sample_pose, (n_dims[j], 1)).T
		template_pose[j]/= div_comp	
	print('galleries')
	print(config.datasets[-1], 'illu', sample_illu, 'pose', sample_pose)


	print('test with Probe C')
	illu_con_matrices = [np.zeros((config.n_illu, config.n_illu)) for j in range(n_feat)]
	pose_con_matrices = [np.zeros((config.n_pose, config.n_pose)) for j in range(n_feat)]
	for usr_label in probe_labels:
		ID = usr_label[0,0]	
		f = os.path.join(dd, 'usr_'+str(ID)+'.txt')
		img = np.array(pickle.load(open(f, 'rb')))
		if len(img.shape)<4: img = img[:, np.newaxis]
		img = img/255.0*2 - 1
		img = torch.Tensor(img).to(device)
		
		if config.evaluate_rec:
			feats = G.forward_enc(G.forward(img))
		else:
			feats = G.forward_enc(img)
			
		for j in range(n_feat):
			v = feats[j].data.cpu().numpy()
			pred_illu = Compare_feat(template_illu[j], v)
			pred_pose = Compare_feat(template_pose[j], v)
			
			for k in range(len(usr_label)):
				illu_con_matrices[j][usr_label[k, 1], pred_illu[k]] += 1
				pose_con_matrices[j][usr_label[k, 2], pred_pose[k]] += 1

	for k in range(n_feat):
		print('feat space', k)
		accu_illu = 0.0
		for j in range(config.n_illu): accu_illu += illu_con_matrices[k][j, j]
		accu_pose =0.0
		for j in range(config.n_pose): accu_pose += pose_con_matrices[k][j, j]
		n1 = np.sum(illu_con_matrices[k])
		print('illu probes', np.sum(illu_con_matrices[k], 1), 'pred_true', accu_illu, 'accuracy rate', accu_illu/n1)
		n2 = np.sum(pose_con_matrices[k])
		print('pose probes', np.sum(pose_con_matrices[k], 1), 'pred_true', accu_pose, 'accuracy rate', accu_pose/n2)

if __name__=='__main__':
	parser = options.TrainOptions()
	config = parser.parse()
	os.environ['CUDA_VISIBLE_DEVICES']=str(config.gpu)
	print(config.name)
	gen_mod = importlib.import_module('models.'+config.net_gen_name)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	imgsize = 128
	G = gen_mod.Generator(config.image_c, config.g_conv_dim, config.dim_attr, config.dim_id, config.dim_source, config.gen_norm)
	G.to(device).eval()
	G.load_state_dict(torch.load(os.path.join(config.networks_dir, config.name, 'models/G.ckpt'), map_location= lambda storage, loc:storage))
	Classify_attr()
	Classify_attr_multipie()
