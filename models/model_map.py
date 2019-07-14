import torch
import torch.nn as nn
import torch.nn.functional as F

class Manipulator(nn.Module):
	"""Generator network."""
	def __init__(self, n_class, feature_dim):
		super(Manipulator, self).__init__()

		self.fc1 = nn.Linear(n_class, feature_dim)
		self.fc2 = nn.Linear(feature_dim*2, feature_dim*2)
		self.fc3 = nn.Linear(feature_dim*2, feature_dim*feature_dim)

	def forward(self, x, c):
		c = self.fc1(c)
		c = torch.cat([x, c], dim=1)
		c = self.fc2(F.relu(c))
		c = self.fc3(F.relu(c))
		c = c.view(x.size(0), x.size(1), x.size(1))
		x = x.unsqueeze(2)
		x = torch.matmul(c, x)
		x = x.view(x.size(0), x.size(1))
		x = F.normalize(x)

		return x
