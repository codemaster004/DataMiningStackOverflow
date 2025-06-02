import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MDN(nn.Module):
	def __init__(self, input_dim, num_mixtures=3, hidden_layers=10, hidden_units=32, dropout=0.1):
		super(MDN, self).__init__()
		
		self.hidden_layers = nn.ModuleList()
		self.hidden_layers.append(nn.Linear(input_dim, hidden_units))
		for _ in range(hidden_layers - 1):
			self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))
		
		self.dropout = nn.Dropout(dropout)
		
		self.num_mixtures = num_mixtures
		self.pi_layer = nn.Linear(hidden_units, num_mixtures)
		self.mu_layer = nn.Linear(hidden_units, num_mixtures)
		self.sigma_layer = nn.Linear(hidden_units, num_mixtures)
	
	def forward(self, x):
		for layer in self.hidden_layers:
			x = F.relu(layer(x))
			x = self.dropout(x)
		
		pi = F.softmax(self.pi_layer(x), dim=1)
		mu = self.mu_layer(x)
		sigma = torch.exp(self.sigma_layer(x))
		return pi, mu, sigma


def mdn_loss(pi, mu, sigma, y):
	'''
	pi: (batch_size, num_mixtures)
	mu: (batch_size, num_mixtures)
	sigma: (batch_size, num_mixtures)
	y: (batch_size,)
	'''
	m = pi.shape[1]
	
	# Expand y to (batch_size, num_mixtures)
	y = y.view(-1, 1).expand_as(mu)
	
	# Compute Gaussian probability for each mixture
	exponent = -0.5 * ((y - mu) / sigma) ** 2
	coeff = 1.0 / (sigma * math.sqrt(2.0 * torch.pi))
	probs = coeff * torch.exp(exponent)
	
	# Weighted sum across mixtures
	weighted_probs = pi * probs
	total_prob = torch.sum(weighted_probs, dim=1)
	
	# Negative log likelihood
	nll = -torch.log(total_prob + 1e-8)
	return torch.mean(nll)
