import random
import torch
import numpy as np

class ImagePool():
	def __init__(self, pool_size):
		self.pool_size = pool_size
		if self.pool_size > 0:
			self.num_imgs = 0
			self.images = []
			self.labels = []

	def query(self, images, labels):
		if self.pool_size == 0:
			return images
		return_images = []
		return_labels = []
		for image, label in zip(images, labels):
			image = torch.unsqueeze(image.data, 0)
			label = label[np.newaxis, :]
			if self.num_imgs < self.pool_size:
				self.num_imgs = self.num_imgs + 1
				self.images.append(image)
				self.labels.append(label)
				return_images.append(image)
				return_labels.append(label)
			else:
				p = random.uniform(0, 1)
				if p > 0.5:
					random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
					tmp = self.images[random_id].clone()
					self.images[random_id] = image
					self.labels[random_id] = label
					return_images.append(tmp)
					return_labels.append(label)
				else:
					return_images.append(image)
					return_labels.append(label)
		return_images = torch.cat(return_images, 0)
		return_labels = np.concatenate(return_labels, 0)
		return return_images, return_labels
