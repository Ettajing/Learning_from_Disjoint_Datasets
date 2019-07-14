import argparse

class TrainOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		# algo
		self.parser.add_argument('--train_type', type= int, default=1, help='network framework [ 0-bridge db, 1-Gen, 2-wgan, 3-bce')
		self.parser.add_argument('--D2', type=bool, default=False, help='attribute classifiers as Discriminator')
		self.parser.add_argument('--D1', type=bool, default=False, help='real/fake Discriminator')
		self.parser.add_argument('--D2_weight', type=bool, default=False, help='use weight while computing Discriminator classifiers loss')
		self.parser.add_argument('--use_pool', type=bool, default=False, help='use of fake image POOL')
		self.parser.add_argument('--use_sigmoid', type=bool, default=False, help='True -BCE Discriminator; False- WGAN Discriminator')
		self.parser.add_argument('--use_cyc', type=bool, default=False, help='use of cyc loss')
		self.parser.add_argument('--homo_loss', type=bool, default=False, help='use of cyc loss')
		self.parser.add_argument('--heter_loss', type=bool, default=False, help='use of cyc loss')
		self.parser.add_argument('--use_triplet', type=bool, default=False, help='use of cyc loss')
		
		# evaluate choise
		self.parser.add_argument('--evaluate_map', type=bool, default=False, help='evaluate the feature space of reconstructed images')
		self.parser.add_argument('--evaluate_rec', type=bool, default=False, help='evaluate the feature space of reconstructed images')
		self.parser.add_argument('--evaluate_map_illu', type=int, default=10, help='evaluate the feature space of reconstructed images')
		self.parser.add_argument('--evaluate_map_pose', type=int, default=10, help='evaluate the feature space of reconstructed images')
		# model parameter  --better not change these parameters
		self.parser.add_argument('--net_gen_name', type=str, default ='model_gen_0', help='G network file ')
		self.parser.add_argument('--net_disc_name', type=str, default ='model_disc_nonorm', help='D network file ')
		
		self.parser.add_argument('--g_conv_dim', type=int, default=16, help='number of conv filters in the first layer of G')
		self.parser.add_argument('--d_conv_dim', type=int, default=16, help='number of conv filters in the first layer of G')
		self.parser.add_argument('--dim_attr', type=int, default=64, help='dimension of pose feature')
		self.parser.add_argument('--dim_id', type=int, default=256, help='dimension of ID feature')
		self.parser.add_argument('--dim_source', type=int, default=64, help='dimension of source feature')
		self.parser.add_argument('--gen_norm', type=str, default='instance', help='dimension of source feature')
		self.parser.add_argument('--attr_alpha', type=float, default=1.8, help='triplet margin for illumination and pose')
		self.parser.add_argument('--source_alpha', type=float, default=2.5, help ='triplet margin for source')
		self.parser.add_argument('--id_alpha', type=float, default= 0.7, help='triplet margin for ID')

		# objectives weights
		self.parser.add_argument('--lambda_disc', type=float, default=0.01, help='lambda for discriminator loss')
		self.parser.add_argument('--lambda_D2', type=float, default=1, help='weight for classifiers loss in discriminator')
		self.parser.add_argument('--lambda_D2_coef', type=float, default=1, help='weight for classifiers loss in discriminator')
		self.parser.add_argument('--lambda_D2_coef_decay', type=float, default=0.005, help='weight for classifiers loss in discriminator')
		self.parser.add_argument('--lambda_gp', type=float, default=1, help='weight for classifiers loss in discriminator')
		self.parser.add_argument('--lambda_triplet', type=float, default=1, help='weight for triplet loss')
		self.parser.add_argument('--lambda_map', type=float, default=1, help='weight for feature maping')
		self.parser.add_argument('--lambda_cyc', type=float, default=1, help='weight for cyc loss')
		self.parser.add_argument('--lambda_cyc_id', type=float, default=1, help='weight for cyc loss')

		# data -- do not change these choices
		self.parser.add_argument('--data_dir', type=str, default='/data/lijing/data_Oct/batch_set2', help='data directories')
		self.parser.add_argument('--datasets', '--list', nargs ='+', default=['caspeal','multipie','cmupie'], help='used datasets')
		self.parser.add_argument('--splits_method', type=int, default=1, help='splits of training and validation dataset, 0 -debug mode')
		self.parser.add_argument('--n_illu', type=int, default=5, help='number of class of illumination')
		self.parser.add_argument('--n_pose', type= int, default=7, help='number of classes of poses')
		self.parser.add_argument('--image_c', type=int, default=1, help='number of image channels')
		# training 			
		self.parser.add_argument('--batch_size', type=int, default=50, help='number of samples in each batch')
		self.parser.add_argument('--patch_size', type=int, default=5, help='number of samples in each batch')
		self.parser.add_argument('--num_epoch', type=int, default=300, help='number of total iterations for training G in each epoch')
		self.parser.add_argument('--g_lr', type=float, default=0.001, help='learning rate for generator')
		self.parser.add_argument('--t_lr', type=float, default=0.001, help='learning rate for attribute manipualator')
		self.parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for attribute manipualator')
		self.parser.add_argument('--g_lr_decay', type=float, default=0.1, help='decay rate for generator')
		self.parser.add_argument('--t_lr_decay', type=float, default=0.01, help='decay learning rate for attribute manipualator')
		self.parser.add_argument('--d_lr_decay', type=float, default=0.1, help='decay learning rate for attribute manipualator')

		self.parser.add_argument('--resume', type=str, default='init_net', help='resume training from this step')
		self.parser.add_argument('--gpu', type=int, default=0)
		self.parser.add_argument('--networks_dir', type=str, default='net_set2_new', help='directory to save models')		
		self.parser.add_argument('--name', type=str, default='test', help='saving model names')

		self.parser.add_argument('--log_step', type=int, default=1)
		self.parser.add_argument('--sample_step', type=int, default=5)
		self.parser.add_argument('--model_save_step', type=int, default=10)
		self.parser.add_argument('--lr_update_step', type=int, default=10)


	def parse(self):
		self.opt = self.parser.parse_args()
		args = vars(self.opt)
		'''
		print('\n--- load options ---')
		for name, value in sorted(args.items()):
		  print('%s: %s' % (str(name), str(value)))
		'''
		# split method
		self.opt.splits = []
		if self.opt.splits_method ==0: # debug
 			self.opt.splits = [[0, 50, 60], [0, 50, 60], [0, 5, 10]] 
		elif self.opt.splits_method==1:
			self.opt.splits = [[0, 200, 230], [0, 200, 300], [0, 20, 60]]

		return self.opt
