import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero,GATConv, GCNConv
import torch_geometric.nn as hetero_nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch_sparse
import tqdm
from torch.nn import BCEWithLogitsLoss
import time
import sys
import pandas as pd
from torch_geometric.loader import LinkNeighborLoader
import torch.optim as optim
from tqdm.auto import trange
import pickle

CSV = r'.\csvfiles'
CKP = r'.\checkpoint'

metadata = (['target_entity', 't_entities', 'aspect', 'a_entities'], [('aspect', 'associated_with', 'a_entities'),('target_entity', 'linked_to', 'aspect'),
	('target_entity', 'associated_with', 't_entities'),
	('a_entities', 'rev_associated_with', 'aspect'),
	('aspect', 'rev_linked_to', 'target_entity'),('t_entities', 'rev_associated_with', 'target_entity')])

class GraphSage(nn.Module):
	def __init__(self, activation, hidden_channel = 128, out_channel= 64):
		super().__init__()
		self.activation = activation
		self.conv1 = SAGEConv((-1, -1), hidden_channel)
		self.conv2 = SAGEConv((-1, -1), out_channel)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x) 
		x = self.conv2(x, edge_index)

		return x

class GraphAttention(nn.Module):
	""" Class to implement Graph Attention Module """

	def __init__(self, activation, hidden_channel = 128 , out_channel = 64, heads = 5):
		super().__init__()
		self.activation = activation
		self.heads = heads
		self.conv1 = GATConv((-1, -1), hidden_channel, add_self_loops = False)
		self.conv2 = GATConv((-1, -1), out_channel, add_self_loops = False)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
		x = self.conv2(x, edge_index)

		return x

class GraphConvolution(nn.Module):
	""" Class to implement Graph Convolution Module """
	def __init__(self, activation, hidden_channel = 128, out_channel = 64):
		super().__init__()
		
		self.activation = activation
		self.conv1 = GCNConv(-1, hidden_channel, add_self_loops = False)
		self.conv2 = GCNConv(-1, out_channel, add_self_loops = False)
		

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
		x = self.conv2(x, edge_index)

		return x

class Classifier(nn.Module):
    def __init__(self, hidden_channel = 128 , hidden_channel2 = 64):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channel, hidden_channel2)
        self.lin2 = torch.nn.Linear(hidden_channel2, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['entity'][row], z_dict['aspect'][col]], dim=-1) #Change to correct aspect and entity names

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
	def __init__(self, output_channel = 300, activation = "relu", aggregation = "mean", encoder_type = "GCN"):
		super().__init__()
		if activation not in ['relu', 'leaky_relu']:
			print('Please provide appropriate activation function.')
		if activation == "leaky_relu":
			activation = nn.LeakyReLU()
		elif activation == 'relu':
			activation = nn.ReLU()
		self.activation = activation

		self.lin1 = hetero_nn.Linear(-1, output_channel)
		self.lin2 = hetero_nn.Linear(-1, output_channel)
		self.encoder_type = encoder_type
		if self.encoder_type == "GAT":
			self.encoder = GraphAttention(activation = self.activation)
		elif self.encoder_type == "GCN":
			self.encoder = GraphConvolution(activation = self.activation)
		elif self.encoder_type == "GSG":
			self.encoder = GraphSage(activation = self.activation)
		else:
			pass

		self.encoder = to_hetero(self.encoder, metadata, aggr=aggregation)

		self.decoder = Classifier()

	def forward(self, x_dict, y_dict, edge_index_dict, edge_label_index):
		for item in x_dict:
			if item == 'target_entity':
				x_dict[item] = self.lin2(torch.cat((x_dict[item], y_dict[item]), dim=1))
			else:
				x_dict[item] = self.lin1(x_dict[item])
		for keys in edge_index_dict:
			print(edge_index_dict[keys].shape)  
		z_dict = self.encoder(x_dict, edge_index_dict)
		return self.decoder(z_dict, edge_label_index)


class Main():
	def __init__(self, graph, mode = "GCN", CKP = "./checkpoints/", model_file = None, device = 'cpu',infer = False, task = 'linkpred'):
		self.device = device
		self.checkpoint = CKP
		self.loss_fn = BCEWithLogitsLoss() 
		self.mode = mode
		self.gnn = Model(encoder_type = self.mode)
		self.task = task

		if infer:
			if model_file is None:
				print('Please provide model file')
				exit()
			else:
				self.gnn.load_state_dict(torch.load(f'{CKP}\\{model_file}'))
	
	def add_negative_edges(self, data, edge_types = ("target_entity", "linked_to", "aspect"), rev_edge_types = ("aspect", "rev_linked_to", "target_entity")):
		transform = T.RandomLinkSplit(
    		num_val = 0,
    		num_test = 0,
    		disjoint_train_ratio = 0.7,   
    		add_negative_train_samples = True,  
    		edge_types = edge_types,
    		rev_edge_types = rev_edge_types
		)
		result, *rest = transform(data)
		return result

	def prepare_dataloader(self, data, batch_size, num_neighbors = [20, 10], edge_type = ("target_entity", "linked_to", "aspect")):
		data = self.add_negative_edges(data)
		edge_label_index = data[edge_type[0], edge_type[1], edge_type[2]].edge_label_index
		edge_label = data["target_entity", "linked_to", "aspect"].edge_label

		data_loader = LinkNeighborLoader(
		    data = data.contiguous(),  
		    num_neighbors = num_neighbors, 
		    edge_label_index = (edge_type, edge_label_index),
		    edge_label = edge_label,
		    batch_size =batch_size,
		    shuffle=True,
		)
		return data_loader

	def evaluate(self, data_loader): 
		total_loss = 0

		for sampled_data in data_loader:
			sampled_data = sampled_data.to(self.device)
			pred = self.gnn(sampled_data.x_dict, sampled_data.y_dict, sampled_data.edge_index_dict, sampled_data['target_entity','linked_to', 'aspect'].edge_label_index)
			edge_label = sampled_data['target_entity','linked_to', 'aspect'].edge_label
			edge_label = edge_label.to(self.device)
			loss = self.loss_fn(pred, edge_label)
			total_loss += loss.float() * pred.numel()

		return total_loss/len(data_loader)

	def train(self, train_loader, val_loader, epochs = 50, optimizer = None, lr_scheduler = None, dataset_type = 'train-small', content = 'sentence'): #Add scaling factor
		best_devloss = sys.maxsize
		loss_val = []
		avg_loss_epoch = []
		train_loss = []
		valid_loss = []
		bestepoch = -99
		model = self.gnn.to(self.device)
		#optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay = weight_decay)

		#pos_weight = torch.tensor([scaling_factor]).to(device)
		#self.loss_fn.pos_weight = pos_weight

		for epoch in trange(1,epochs + 1):
			total_loss = total_examples = 0
			
			self.gnn.train()
			self.gnn.to(self.device)
			if epoch == 1:
				print('Epoch %d / %d : Train Loss = %.6f, Val Loss = %.6f'
					% (epoch, epochs, self.evaluate(train_loader), self.evaluate(val_loader)))

			for sampled_data in train_loader:
				optimizer.zero_grad()
				sampled_data = sampled_data.to(self.device)
				pred = model(sampled_data.x_dict, sampled_data.y_dict, sampled_data.edge_index_dict, sampled_data['target_entity', 'aspect'].edge_label_index)
				edge_label = sampled_data['target_entity', 'aspect'].edge_label
				edge_label = edge_label.to(self.device)
				loss = self.loss_fn(pred, edge_label)
				loss.backward()
				optimizer.step()
				total_loss += float(loss) * pred.numel()
				with torch.no_grad():
					self.gnn.eval()
					trainloss = self.evaluate(train_loader)
					valloss = self.evaluate(val_loader)
					if valloss < best_devloss:
						best_devloss = valloss
						bestepoch = epoch
						torch.save(self.gnn.state_dict(),
				                       f"{CKP}\\{dataset_type}_{content}_{self.task}_{self.mode}.pt")
					if lr_scheduler is not None:
						lr_scheduler.step(valloss)

					train_loss.append(trainloss)
					valid_loss.append(valloss)

					print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, trainloss, valloss))

				if epoch - bestepoch >= 10:
					print("Early stopping")
					break

		print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, self.evaluate(train_loader), self.evaluate(val_loader)))


	def predict(self, test_loader, root_csv, CSV = CSV):
		#pos_weight = torch.tensor(scaling_factor)
		self.gnn.eval()
		total_loss = 0	
		preds = []
		ground_truths = []
		#self.loss_fn.pos_weight = pos_weight

		with torch.no_grad():
			for sampled_data in tqdm.tqdm(train_loader):
				sampled_data = sampled_data.to(self.device)
				y_pred = self.gnn(sampled_data)
				test_loss = loss_fn(pred, sampled_data['target_entity', 'aspect'].edge_label)
				preds.append(torch.sigmoid(self.gnn(sampled_data)))
				ground_truths.append(sampled_data['target_entity', 'aspect'].edge_label)
				total_loss += test_loss


		pred = torch.cat(preds, dim=0).numpy() # Joins the two tensors
		ground_truth = torch.cat(ground_truths, dim=0).numpy()

		y_pred_class = [1 if el > 0.5 else 0 for el in pred]

		f1_sc = multiclass_f1_score(pred, y_pred_class, num_classes = 2)
		avg_loss = total_loss / len(test_loader)

		print('Test loss is %0.6f' % avg_loss)
		print('Test F1 score is %0.6f' % avg_f1_sc)


def run_gnn(device, CKP, mode, train = True, train_graph = None, val_graph = None, test_graph = None, model_file = None, epochs = 5, lr = 0.001, weight_decay = 1e-4, CSV = None, root_csv = None, batch_size = None, dataset_type = 'train-small', content = 'sentence', task = 'linkpred'):
	start = time.time()
	if not train:
		if root_csv is None:
			print('Please provide a root csv file')
			exit()
		mainclass = Main(graph = test_graph, CKP = CKP, device = device, infer = True, model_file = model_file, task = task)
		test_loader, scaling_factor = prepare_testdataset(data, batch_size = batch_size)
		mainclass.predict(test_data, root_csv, CSV = CSV)
	else: 
		mainclass = Main(graph = train_graph, mode = mode, CKP = CKP, device = device, task = task)
		train_loader = mainclass.prepare_dataloader(train_graph, batch_size = batch_size)
		sampled_data = next(iter(train_loader))
		print(sampled_data)
		val_loader = mainclass.prepare_dataloader(val_graph, batch_size = batch_size)
		weight_decay = 1e-4
		opt = optim.Adam(mainclass.gnn.parameters(), lr = lr, weight_decay = weight_decay)
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 4)
		mainclass.train(train_loader, val_loader, epochs, opt, lr_scheduler)

	end = time.time()
	if train:
		print(f'Total time taken train the Graph Neural Network is {end - start} seconds')

	else:
		print(f'Total time taken test the Graph Neural Network is {end - start} seconds')