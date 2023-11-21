import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import trange
import torch.optim as optim
from statistics import mean
import sys
import pickle
PTH = 'C:\\Users\\Neela\\Documents\\GitHub\\EntityAspectLinking\\picklefiles'
CKP = 'C:\\Users\\Neela\\Documents\\GitHub\\EntityAspectLinking\\checkpoint\\'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Network(nn.Module):
	def __init__(self, inp, hidden_1 = 256, hidden_2 = 128, hidden_3 = 32, device = None):
		super(Network, self).__init__()
		self.model = nn.Sequential(nn.Linear(inp, hidden_1, bias = True),  nn.LeakyReLU(),
			nn.Linear(hidden_1, hidden_2, bias = True), nn.LeakyReLU(), nn.Linear(hidden_2, hidden_3, bias = True),
			nn.Linear(hidden_3, 1))
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			module.weight = torch.nn.parameter.Parameter(
                nn.init.kaiming_normal_(module.weight,
                                        mode="fan_in",
                                        nonlinearity='relu')*0.1)
			if module.bias is not None:
				nn.init.constant_(module.bias, 0.01)
	
	def forward(self, x):
		#print(x.size())
		pred = self.model(x)
		return pred

def prepare_dataset(file, train_ratio, val_ratio, batch_size,):
	with open(f'{PTH}\\{file}', 'rb') as f:
		data = pickle.load(f)
	f.close()
	
	size = data.shape[0]
	train_size = int(train_ratio * size)
	val_size = int(val_ratio * size)
	test_size = size - train_size - val_size
	data = torch.tensor(data)	
	features = data[ : , : -1]
	#print(features.shape)
	target = data[:, -1]
	#print(len(target))
	data = TensorDataset(features, target)	
	train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
	val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
	test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
	
	return train_loader, val_loader, test_loader

def evaluate(dataloader, model, loss):
	runningloss = 0
	for data in dataloader:
		x, t = data
		x = x.to(device, dtype = torch.float32)
		t = t.to(device, dtype = torch.float32)
		#print(x.size())
		pred = model(x)
		pred = pred.view(t.shape)
		ls = loss(pred, t)
		runningloss += ls.item()
	return runningloss / len(dataloader) 
		


def train(train_ratio, val_ratio, batch_size, epochs, lr = 0.001, weight_decay = 1e-4):
	best_devloss = sys.maxsize
	loss_val = []
	avg_loss_epoch = []
	train_loss = []
	valid_loss = []
	loss_fn = nn.BCEWithLogitsLoss()
	train_loader, val_loader, test_loader = prepare_dataset('random_dataset.pkl', train_ratio, val_ratio, batch_size)
	model = Network(584, device = device)
	opt = optim.Adam(model.parameters(),
                         lr=lr, weight_decay=weight_decay)
	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=4)
	for epoch in trange(epochs):
		if epoch == 0:
			print('Epoch %d / %d : Train Loss = %.6f, Val Loss = %.6f, Test Loss = %.6f'
				% (epoch, epochs, evaluate(train_loader, model, loss_fn), evaluate(val_loader, model, loss_fn), evaluate(test_loader, model, loss_fn)))

		start = len(loss_val)
		for i, data in enumerate(train_loader, 0):
			x, y = data
			x = x.to(device, dtype = torch.float32)
			y = y.to(device, dtype = torch.float32)
			y_pred = model(x)
			y_pred = y_pred.view(y.shape)
			opt.zero_grad()
			loss = loss_fn(y_pred, y)
			loss.backward()
			opt.step()
			loss_val.append(loss.item())
		end = len(loss_val)
		avg_loss_epoch.append(mean(loss_val[start : end + 1]))
		with torch.no_grad():
			trainloss = evaluate(train_loader, model, loss_fn)
			valloss = evaluate(train_loader, model, loss_fn)
			if valloss < best_devloss:
				best_devloss = valloss
				bestepoch = epoch
				torch.save(model.state_dict(),
                               f"{CKP}DNN_EAL_rand.pt")
			if lr_scheduler is not None:
				lr_scheduler.step(valloss)
				
			train_loss.append(trainloss)
			valid_loss.append(valloss)

			print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, trainloss, valloss))


		if epoch - bestepoch >= 10:
			print("Early stopping")
			break

	print('Epoch: %d/%d, Train loss: %0.6f, Valid loss: %0.6f, Test loss:%0.6f' %
		(epoch, epochs, evaluate(train_loader, model, loss_fn), evaluate(val_loader, model, loss_fn), evaluate(test_loader, model, loss_fn)))


		


	

'''train_loader, _, _ = prepare_dataset('random_dataset.pkl', 0.6, 0.2, 128)
for data in train_loader:
	x, y = data
	x = x.to(device)
	print(x.size())'''
train(0.6, 0.1, 32, 200)