import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torcheval.metrics.functional import multiclass_f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange
import torch.optim as optim
from statistics import mean
import sys
import pickle
import time

CKP = r'.\checkpoint'
CSV = r'.\csvfiles'

class EALNetwork(nn.Module):

	def __init__(self, inp, hidden_1 = 256, hidden_2 = 128, hidden_3 = 64, hidden_4 = 32, device = 'cpu'):

		super(Network, self).__init__()
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
		self.model.to(device)
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
	def __init__(self, infer = False, input_dim = 684, CKP = CKP, model_file = None, device = 'cpu', pos_weight = 1):
		self.CKP = CKP
		self.device = device
		self.dnn = EALNetwork(input_dim, device = device)
		self.loss_fn = BCEWithLogitsLoss()
		
		if infer:
			if model_file is None:
				print('Please provide model file')
				exit()
			else:
				self.dnn.load_state_dict(torch.load(f'{CKP}\\{model_file}'))
	


	def prepare_traindataset(self, data, td = True, batch_size = 128):
		features = data[:, : -1]
		target = data[:, -1]
		sc = StandardScaler()
		features = sc.fit_transform(features)
		dataset = TensorDataset(torch.tensor(features), torch.tensor(target))
		if train == True:
			neg_count = 0
			for el in target:
				if el == 0:
					neg_count += 1
			pos_count = len(target) - neg_count
			scaling_factor = round(neg_count / pos_count, 2)
			train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
			return train_loader, scaling_factor
		else:
			data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
			return data_loader

	def prepare_testdataset(self, data):
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
		test_dataset = Dataset(torch.tensor(features, dtype = 'float32'), torch.tensor(target))
		test_loader = DataLoader(test_dataset, batch_size = batch_size)
		return test_loader, scaling_factor

	def evaluate(self, dataloader, model, loss_fn):
		runningloss = 0
		for data in dataloader:
			x, t = data
			x = x.to(device, dtype = torch.float32)
			t = t.to(device, dtype = torch.float32)
			pred_logit = model(x).to(device)
			pred_logit = pred.view(t.shape)
			ls = loss_fn(pred, t)
			runningloss += ls.item()
		return runningloss / len(dataloader)

	def train(self, train_loader, val_loader, epochs, opt, lr_scheduler = None, scaling_factor = 1):
		best_devloss = sys.maxsize
		loss_val = []
		avg_loss_epoch = []
		train_loss = []
		valid_loss = []
		pos_weight = torch.tensor([scaling_factor]).to(device)
		self.loss_fn.pos_weight = pos_weight
		for epoch in trange(epochs):
			self.dnn.train()
			self.dnn.to(device)
			if epoch == 0:
				print('Epoch %d / %d : Train Loss = %.6f, Val Loss = %.6f'
					% (epoch, epochs, evaluate(train_loader, self.dnn, loss_fn), evaluate(val_loader, self.dnn, loss_fn)))

			start = len(loss_val)
			for i, data in enumerate(train_loader, 0):
				x, y = data
				x = x.to(device, dtype = torch.float32)
				y = y.to(device, dtype = torch.float32)
				y_pred = self.dnn(x)
				y_pred = y_pred.view(y.shape)
				opt.zero_grad()
				pos_weight = torch.tensor(scaling_factor)
				loss = loss_fn(y_pred, y)
				loss.backward()
				opt.step()
				loss_val.append(loss.item())
			end = len(loss_val)
			avg_loss_epoch.append(mean(loss_val[start : end + 1]))
			with torch.no_grad():
				self.dnn.eval()
				trainloss = evaluate(train_loader, self.dnn, loss_fn)
				valloss = evaluate(val_loader, self.dnn, loss_fn)
				if valloss < best_devloss:
					best_devloss = valloss
					bestepoch = epoch
					torch.save(self.dnn.state_dict(),
	                               f"{CKP}\\{mode}.pt")
				if lr_scheduler is not None:
					lr_scheduler.step(valloss)
					
				train_loss.append(trainloss)
				valid_loss.append(valloss)

				print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, trainloss, valloss))


			if epoch - bestepoch >= 10:
				print("Early stopping")
				break

		print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' %
			(epoch, epochs, evaluate(train_loader, self.dnn, loss_fn), evaluate(val_loader, self.dnn, loss_fn)))


	def get_test_id(self, root_csv, pred_class, pred_prob):	
		test_id = pd.read_csv(f'{CSV}\\{root_csv}')
		test_id['Predicted Label'] = pred_class
		test_id['Predicted Probabilities'] = pred_prob
		test_id.to_csv(f'{CSV}\\{self.model_file}_pred.csv', index = False)
		print('Predictions saved successfully')
		return

	def predict(self, data, root_csv, CSV = CSV):
		test_loader, scaling_factor = prepare_testdataset(data)
		pos_weight = torch.tensor(scaling_factor)
		self.dnn.eval()
		y_pred_tot = []
		y_prob_tot = []
		with torch.no_grad():
			for batch in test_loader:
				x, y = batch
				x = x.to(device)
				y = y.to(device)
				y_pred = self.dnn(x)
				self.loss_fn.pos_weight = pos_weight
				test_loss = loss_fn(y, y_pred)
				y_pred_prob = torch.sigmoid(y_pred)
				y_pred_class = [1 if el > 0.5 else 0 for el in y_pred_prob]
				y_pred_class = torch.tensor(y_pred_class).to(device)
				f1_sc = multiclass_f1_score(y_pred_class, y, num_classes = 2)
		print('Test loss is %0.6f' % test_loss)
		print('Test F1 score is %0.6f' % f1_sc)
		get_test_id(root_csv, pred_class, pred_prob)


def run_dnn(device, CKP, train = True, train_data = None, val_data = None, test_data = None, model_file = None, epochs = 5, lr = 0.001, weight_decay = 1e-4, CSV = None, root_csv = None, batch_size = None):
	start = time.time()
	if not train:
		if root_csv is None:
			print('Please provide a root csv file')
			exit()
		mainclass = Main(CKP = CKP, device = device, infer = True, model_file = model_file)
		mainclass.predict(test_data, root_csv, CSV = CSV)
	else:
		mainclass = Main(CKP = CKP, device = device)
		train_loader, scaling_factor = mainclass.prepare_traindataset(train_data, batch_size = batch_size, device = device)
		val_loader = mainclass.prepare_traindataset(val_data, td = False, batch_size = batch_size)
		opt = optim.Adam(mainclass.dnn.parameters(), lr = lr, weight_decay = weight_decay)
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 4)
		mainclass.train(train_loader, val_loader, epochs, opt, lr_scheduler, scaling_factor = scaling_factor)

	end = time.time()
	if train:
		print(f'Total time taken train the Deep Neural Network is {end - start} seconds')

	else:
		print(f'Total time taken test the Deep Neural Network is {end - start} seconds')



