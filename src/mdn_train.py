import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mdn import MDN, mdn_loss

# Load CSV
df = pd.read_csv('../data/train.csv')

# Columns
numeric_features = []
categorical_features = ['Professional', 'Country', 'WorkLang', 'CompanyType']
target_col = 'SalaryUSD'

# Encode categorical features
label_encoders = {}
for col in categorical_features:
	le = LabelEncoder()
	df[col] = le.fit_transform(df[col])
	label_encoders[col] = le  # Save encoders if needed later

# Combine all features
all_features = numeric_features + categorical_features
X = df[all_features].values.astype('float32')
y = df[target_col].values.astype('float32')


# Custom Dataset
class CSVDataset(Dataset):
	def __init__(self, features, targets):
		self.X = torch.tensor(features)
		self.y = torch.tensor(targets)
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


# Create dataset
dataset = CSVDataset(X, y)

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create MDN model
input_dim = X.shape[1]
num_mixtures = 3
model = MDN(input_dim, num_mixtures)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
	total_loss = 0.0
	for batch_X, batch_y in dataloader:
		optimizer.zero_grad()
		pi, mu, sigma = model(batch_X)
		loss = mdn_loss(pi, mu, sigma, batch_y)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	
	print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
