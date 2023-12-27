import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from torcheval.metrics.functional import multiclass_f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange
import torch.optim as optim
from statistics import mean
import sys
import pickle
import pandas as pd
import time

CKP = r'.\checkpoint'
CSV = r'.\csvfiles'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EALNetwork(nn.Module):

	def __init__(self, inp, hidden_1 = 256, hidden_2 = 128, hidden_3 = 64, hidden_4 = 32, device = device):

		super(EALNetwork, self).__init__()
		self.device = device
		self.model = nn.Sequential(nn.Linear(inp, hidden_1, bias = True),  
			nn.LeakyReLU(),
			nn.Dropout(0.8),
			nn.Linear(hidden_1, hidden_2, bias = True), 
			nn.LeakyReLU(), 
			nn.Dropout(0.8),
			nn.Linear(hidden_2, hidden_3, bias = True),
			nn.LeakyReLU(),
			nn.Linear(hidden_3, hidden_4, bias = True),
			nn.LeakyReLU(),
			nn.Dropout(0.9),
			nn.Linear(hidden_4, 1))
		self.model.to(self.device)
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
		pred = self.model(x)
		return pred


class Main():
	def __init__(self, infer = False, input_dim = 684, CKP = CKP, model_file = None, device = device, pos_weight = 1):
		self.CKP = CKP
		self.device = device
		self.dnn = EALNetwork(input_dim, device = self.device)
		self.loss_fn = nn.BCEWithLogitsLoss()
		
		if infer:
			if model_file is None:
				print('Please provide model file')
				exit()
			else:
				self.dnn.load_state_dict(torch.load(f'{self.CKP}\\{model_file}.pt'))
	


	def prepare_traindataset(self, data, td = True, batch_size = 128):
		features = data[:, : -1]
		target = data[:, -1]
		sc = StandardScaler()
		features = sc.fit_transform(features)
		dataset = TensorDataset(torch.tensor(features), 
			torch.tensor(target))
		neg_count = 0
		for el in target:
			if el == 0:
				neg_count += 1

		pos_count = len(target) - neg_count
		scaling_factor = round(neg_count / pos_count, 2)
		count = [neg_count, pos_count]
		weights = 1 / torch.tensor(count, dtype=torch.float32)
		sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
		dataloader = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
		if td == True:
			return dataloader, scaling_factor
		else:
			return dataloader

	def prepare_merged_traindataset(self, train, val, train_ratio = 0.8, val_ratio = 0.2, batch_size = 128):
		data = torch.cat((train, val), dim = 0)
		features = data[ : , : -1]
		sc = StandardScaler()
		features = sc.fit_transform(features)
		target = data[:, -1]
		train_data = TensorDataset(torch.tensor(features), torch.tensor(target))
		neg_count = 0
		for el in target:
			if el == 0:
				neg_count += 1
		pos_count = len(target) - neg_count
		size = features.shape[0]
		train_size = int(train_ratio * size)
		val_size = int(val_ratio * size)
		train_size += size - train_size - val_size
		train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
		train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
		val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
		return train_loader, val_loader, neg_count / pos_count
	
	def prepare_testdataset(self, data, batch_size = 1):
		features = data[ :, : -1]
		target = data[:, -1]
		sc = StandardScaler()
		features = sc.fit_transform(features)
		neg_count = 0
		for el in target:
			if el == 0:
				neg_count += 1
		pos_count = len(target) - neg_count
		scaling_factor = round(neg_count / pos_count, 2)
		test_dataset = TensorDataset(torch.tensor(features, dtype = torch.float32), torch.tensor(target))
		test_loader = DataLoader(test_dataset, batch_size = batch_size)
		return test_loader, scaling_factor

	def evaluate(self, dataloader):
		runningloss = 0
		for data in dataloader:
			x, t = data
			x = x.to(self.device, dtype = torch.float32)
			t = t.to(self.device, dtype = torch.float32)
			pred = self.dnn(x).to(self.device)
			pred = pred.view(t.shape)
			ls = self.loss_fn(pred, t)
			runningloss += ls.item()
		return runningloss / len(dataloader)

	def train(self, train_loader, val_loader, epochs, opt, lr_scheduler = None, scaling_factor = 1, dataset_type = 'train-small', content = 'sentence'):
		best_devloss = sys.maxsize
		loss_val = []
		avg_loss_epoch = []
		train_loss = []
		valid_loss = []
		pos_weight = torch.tensor([scaling_factor]).to(self.device)
		self.loss_fn.pos_weight = pos_weight
		print(self.loss_fn)
		print('Training the Deep Neural Network......')
		count = 0
		for epoch in trange(epochs):
			self.dnn.train()
			self.dnn.to(self.device)
			if epoch == 0:
				print('Epoch %d / %d : Train Loss = %.6f, Val Loss = %.6f'
					% (epoch, epochs, self.evaluate(train_loader), self.evaluate(val_loader)))

			start = len(loss_val)
			for i, data in enumerate(train_loader, 0):
				x, y = data
				x = x.to(self.device, dtype = torch.float32)
				y = y.to(self.device, dtype = torch.float32)
				y_pred = self.dnn(x)
				y_pred = y_pred.view(y.shape)
				opt.zero_grad()
				pos_weight = torch.tensor(scaling_factor)
				loss = self.loss_fn(y_pred, y)
				loss.backward()
				opt.step()
				loss_val.append(loss.item())
			end = len(loss_val)
			#print(mean(loss_val[start : end + 1]))
			avg_loss_epoch.append(mean(loss_val[start : end + 1]))
			with torch.no_grad():
				self.dnn.eval()
				trainloss = self.evaluate(train_loader)
				valloss = self.evaluate(val_loader)
				if valloss < best_devloss:
					best_devloss = valloss
					bestepoch = epoch
					torch.save(self.dnn.state_dict(),
	                               f"{self.CKP}\\{dataset_type}_{content}_dnn.pt")
				if lr_scheduler is not None:
					lr_scheduler.step(valloss)
					
				train_loss.append(trainloss)
				valid_loss.append(valloss)

				print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, avg_loss_epoch[epoch], valloss))


			if epoch - bestepoch >= 10:
				print("Early stopping")
				break

		print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' %
			(epoch, epochs, self.evaluate(train_loader), self.evaluate(val_loader)))
		print(avg_loss_epoch)


	def get_test_id(self, root_csv, pred_class, pred_prob, content = 'sentence', tag = 'test', CSV = CSV):	
		test_id = pd.read_csv(f'{CSV}\\{root_csv}')
		test_id['Predicted Label'] = pred_class
		test_id['Predicted Probabilities'] = pred_prob
		test_id.to_csv(f'{CSV}\\{tag}_{content}_dnn_pred.csv', index = False)
		print('Predictions saved successfully')
		return

	def predict(self, CSV, test_loader, root_csv, content, tag, scaling_factor = 1):
		pos_weight = torch.tensor(scaling_factor)
		self.dnn.eval()
		pred_class_tot = []
		pred_prob_tot = []
		tot_f1_sc = 0
		total_loss = 0
		print('Loading the model file for predictions')
		with torch.no_grad():
			for batch in test_loader:
				x, y = batch
				x = x.to(self.device)
				y = y.to(self.device)
				y_pred = self.dnn(x)
				y_pred = y_pred.view(y.shape[0])
				self.loss_fn.pos_weight = pos_weight
				test_loss = self.loss_fn(y, y_pred)
				y_pred_prob = torch.sigmoid(y_pred)
				y_pred_prob = torch.squeeze(y_pred_prob)

				pred_prob_tot.extend(y_pred_prob.cpu().tolist())
				y_pred_class = [1 if el > 0.5 else 0 for el in y_pred_prob]
				pred_class_tot.extend(y_pred_class)
				y_pred_class = torch.tensor(y_pred_class).to(self.device)
				f1_sc = multiclass_f1_score(y_pred_class, y, num_classes = 2)
				tot_f1_sc += f1_sc
				total_loss += test_loss
		avg_loss = total_loss / len(test_loader)
		avg_f1_sc = tot_f1_sc / len(test_loader)
		print('Test loss is %0.6f' % avg_loss)
		print('Test F1 score is %0.6f' % avg_f1_sc)
		self.get_test_id(root_csv, pred_class_tot, pred_prob_tot, content = content, tag = tag, CSV = CSV)


def run_dnn(device, CKP, dataset_type = 'train-small', content = 'sentence', train_ratio = None, val_ratio = None, train = True, train_data = None, 
	val_data = None, test_data = None, model_file = None, epochs = 5, lr = 0.001, weight_decay = 1e-4, CSV = None, root_csv = None, batch_size = None):
	start = time.time()
	if not train:
		if root_csv is None:
			print('Please provide a root csv file')
			exit()
		mainclass = Main(CKP = CKP, device = device, infer = True, model_file = model_file)
		if test_data is None:
			print('Please provide test data')
			exit()
		test_loader, scaling_factor = mainclass.prepare_testdataset(data = test_data, batch_size = batch_size)
		mainclass.predict(CSV, test_loader, root_csv, content = content, tag = dataset_type, scaling_factor = scaling_factor)
	else:
		mainclass = Main(CKP = CKP, device = device)
		train_loader, scaling_factor = mainclass.prepare_traindataset(train_data, batch_size = batch_size)
		'''train_loader, val_loader, scaling_factor = mainclass.prepare_merged_traindataset(train = train_data, val = val_data,
		batch_size = batch_size, train_ratio = train_ratio, val_ratio = val_ratio)'''
		val_loader = mainclass.prepare_traindataset(val_data, td = False, batch_size = batch_size)
		opt = optim.Adam(mainclass.dnn.parameters(), lr = lr, weight_decay = weight_decay)
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 4)
		mainclass.train(train_loader, val_loader, epochs, opt, lr_scheduler, scaling_factor = scaling_factor, dataset_type = dataset_type, content = content)


	end = time.time()
	
	if train:
		print(f'Total time taken train the Deep Neural Network is {end - start} seconds')

	else:
		print(f'Total time taken test the Deep Neural Network is {end - start} seconds')



