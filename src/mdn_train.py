import torch
from torch.utils.data import Dataset



# Custom Dataset
class CSVDataset(Dataset):
	def __init__(self, features, targets):
		self.X = torch.tensor(features)
		self.y = torch.tensor(targets)
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

