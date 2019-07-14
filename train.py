import os
import argparse
from torch.backends import cudnn
import torch
import importlib

from source import solver
from source import options


def main(config):
	cudnn.benchmark = True
	os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

	torch.manual_seed(1337)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(1337)
	# create directories
	config.model_dir = os.path.join(config.networks_dir, config.name)
	if not os.path.exists(config.model_dir):
		os.makedirs(config.model_dir)
	config.log_dir = os.path.join(config.model_dir, 'logs')
	if not os.path.exists(config.log_dir):
		os.makedirs(config.log_dir)
	config.model_save_dir = os.path.join(config.model_dir, 'models')
	if not os.path.exists(config.model_save_dir):
		os.makedirs(config.model_save_dir)
	config.sample_dir = os.path.join(config.model_dir, 'samples')
	if not os.path.exists(config.sample_dir):
		os.makedirs(config.sample_dir)

	# solver
	solution = solver.Solver(config)
	solution.train()
	
if __name__ == '__main__':
	parser = options.TrainOptions()
	config = parser.parse()
	main(config)
