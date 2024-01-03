import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero,GATConv, GCNConv
import torch_geometric.nn as hetero_nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch_sparse
import tqdm
from torch.nn import BCEWithLogitsLoss
from torcheval.metrics.functional import multiclass_f1_score
import time
import sys
import pandas as pd
from torch_geometric.loader import LinkNeighborLoader
import torch.optim as optim
from tqdm.auto import trange
import pickle
from statistics import mean
import graphutils
import utils
from transformers import AutoConfig, AutoModel, AutoTokenizer

torch.cuda.empty_cache()
CSV = r'.\csvfiles'
CKP = r'.\checkpoint'

metadata = ['target_entity', 'aspect', 't_entities', 'a_entities'], [('target_entity', 'linked_to', 'aspect'), ('target_entity', 'associated_to', 't_entities'), ('aspect', 'associated_to', 'a_entities'), ('aspect', 'rev_linked_to', 'target_entity'), ('t_entities', 'rev_associated_to', 'target_entity'), ('a_entities', 'rev_associated_to', 'aspect')]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Bert(nn.Module):
	def __init__(self, pretrained):
		super().__init__()
		self._pretrained = pretrained
		self._config = AutoConfig.from_pretrained(self._pretrained)
		self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained)
		self.bert = AutoModel.from_pretrained(self._pretrained, config = self._config)

	def _freeze_bert(self):
		print('Freezing all but the last layer of BERT....')
		for param in self.bert.parameters():
			param.requires_grad = False
		for param in self.bert.encoder.layer[-1].parameters():
			param.requires_grad = True
		print('Successfully freezed BERT.')

	def _preprocess(self, text):
		encoded_dict = self._tokenizer.encode_plus(text = text, add_special_tokens = True, max_length = 512, padding = 'max_length', truncation = True, return_attention_mask = True, return_tensors = 'pt')
		encoded_dict = {key: value.to(device) for key, value in encoded_dict.items()}
		return encoded_dict

		#return encoded_dict['input_ids'], encoded_dict['attention_mask']


	def forward(self, encoded_dict):
		output = self.bert(**encoded_dict)
		return output.last_hidden_state[:, 0, :]


class GraphSage(nn.Module):
	def __init__(self, activation, hidden_channel = 128, out_channel= 64):
		super().__init__()
		self.activation = activation
		self.conv1 = SAGEConv((-1, -1), hidden_channel, normalize = True, drop_out = 0.8)
		self.conv2 = SAGEConv((-1, -1), out_channel, normalize = True, drop_out = 0.8)

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
		self.conv1 = GATConv((-1, -1), hidden_channel, add_self_loops = False, drop_out = 0.8, v2 = True)
		self.conv2 = GATConv((-1, -1), out_channel, add_self_loops = False, drop_out = 0.8, v2 = True)

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
		

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
		x = self.conv2(x, edge_index)

		return x

class Classifier(nn.Module):
    def __init__(self, hidden_channel = 128, hidden_channel2 = 64):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channel, hidden_channel2)
        self.lin2 = torch.nn.Linear(hidden_channel2, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['target_entity'][row], z_dict['aspect'][col]], dim=-1) #Change to correct aspect and entity names

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(nn.Module):
	def __init__(self, lm, output_channel = 768, activation = "relu", aggregation = "mean", encoder_type = "GSG"):
		super().__init__()
		if activation not in ['relu', 'leaky_relu']:
			print('Please provide appropriate activation function.')
		if activation == "leaky_relu":
			activation = nn.LeakyReLU()
		elif activation == 'relu':
			activation = nn.ReLU()
		self.activation = activation
		self.bert = Bert(pretrained = lm)
		self.bert._freeze_bert()
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

	def forward(self, feature_dict, x_dict, y_dict, edge_index_dict, edge_label_index, mode):
		for item in x_dict:
			x_feat = torch.zeros(len(x_dict[item]), 768).to(device)
			if item == 'target_entity':
				y_feat = torch.zeros(len(x_dict[item]), 768).to(device)
				for el in range(len(x_dict[item])):
					x_encoded_dict = self.bert._preprocess(feature_dict[mode][item]['x'][int(x_dict[item][el].cpu())])
					y_encoded_dict = self.bert._preprocess(feature_dict[mode][item]['y'][int(x_dict[item][el].cpu())])
					bert_xoutput = self.bert(x_encoded_dict)
					bert_youtput = self.bert(y_encoded_dict)
					x_feat[el] = bert_xoutput
					y_feat[el] = bert_youtput
				x_dict[item] = x_feat
				y_dict[item] = y_feat
				x_dict[item] = self.lin2(torch.cat((x_dict[item], y_dict[item]), dim=1))
			else:
				for el in range(len(x_dict[item])):
					encoded_dict = self.bert._preprocess(feature_dict[mode][item][int(x_dict[item][el].cpu())])
					bert_xoutput = self.bert(encoded_dict)
					x_feat[el] = bert_xoutput
				x_dict[item] = x_feat
			z_dict = self.encoder(x_dict, edge_index_dict)
		return self.decoder(z_dict, edge_label_index)


class Main():
	def __init__(self, feature_dict, lm, graph, device, mode = "GCN", CKP = "./checkpoint", model_file = None, infer = False, task = 'linkpred'):
		self.device = device
		self.checkpoint = CKP
		self.loss_fn = BCEWithLogitsLoss() 
		self.mode = mode
		self.ft_gnn = Model(encoder_type = self.mode, lm = lm)
		self.task = task
		self.feature_dict = feature_dict

		if infer:
			if model_file is None:
				print('Please provide model file')
				exit()
			else:
				self.ft_gnn.load_state_dict(torch.load(f'{self.checkpoint}\\{model_file}'))
	
	def add_negative_edges(self, data, edge_types = ("target_entity", "linked_to", "aspect"), rev_edge_types = ("aspect", "rev_linked_to", "target_entity"), dtr = 0.999, nsr = 1):

		transform = T.RandomLinkSplit(
	    		num_val = 0,
	    		num_test = 0,
	    		disjoint_train_ratio = dtr,   
	    		add_negative_train_samples = True,  
	    		edge_types = edge_types,
	    		rev_edge_types = rev_edge_types,
	    		is_undirected = True,
	    		neg_sampling_ratio = nsr
			)

		result, *rest = transform(data)
		return result

	def prepare_dataloader(self, data, dtr, nsr, batch_size, tag = 'train', num_neighbors = [20, 10], edge_type = ("target_entity", "linked_to", "aspect")):
		if tag in ['val', 'test']:
			data = self.add_negative_edges(data, dtr = dtr, nsr = nsr)
		elif tag == 'train':
			data = self.add_negative_edges(data, dtr = dtr, nsr = nsr)
		else:
			print('Please provide appropriate tag type')
			exit()
		edge_label_index = data[edge_type[0], edge_type[1], edge_type[2]].edge_label_index
		edge_label = data[edge_type[0], edge_type[1], edge_type[2]].edge_label
		data_loader = LinkNeighborLoader(
		    data = data,  
		    num_neighbors = num_neighbors, 
		    edge_label_index = (edge_type, edge_label_index),
		    edge_label = edge_label,
		    batch_size = batch_size,
		    shuffle = True,
		)
		return data_loader

	def evaluate(self, tag, data_loader): 
		total_loss = 0

		for sampled_data in data_loader:
			sampled_data = sampled_data.to(self.device)
			#print(sampled_data)
			pred = self.ft_gnn(self.feature_dict, sampled_data.x_dict, sampled_data.y_dict, sampled_data.edge_index_dict, sampled_data['target_entity','linked_to', 'aspect'].edge_label_index, tag)
			edge_label = sampled_data['target_entity','linked_to', 'aspect'].edge_label
			edge_label = edge_label.to(self.device)
			loss = self.loss_fn(pred, edge_label)

			print(loss)
			total_loss += loss.item()
		return total_loss/len(data_loader)

	def train(self, train_loader, val_loader, epochs = 50, optimizer = None, lr_scheduler = None, dataset_type = 'train-small', content = 'sentence'): #Add scaling factor
		best_devloss = sys.maxsize
		loss_val = []
		avg_loss_epoch = []
		train_loss = []
		valid_loss = []
		bestepoch = -99
		self.ft_gnn.to(self.device)
		#optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay = weight_decay)

		#pos_weight = torch.tensor([scaling_factor]).to(device)
		#self.loss_fn.pos_weight = pos_weight
		print('Training starting...')
		for epoch in trange(1,epochs + 1):
			
			total_loss = total_examples = 0
			self.ft_gnn.train()
			if epoch == 1:
				print('Epoch %d / %d : Train Loss = %.6f, Val Loss = %.6f'
					% (epoch, epochs, self.evaluate(data_loader = train_loader, tag = 'train'), self.evaluate(data_loader = val_loader, tag = 'val')))
			start = len(loss_val)
			for sampled_data in train_loader:
				optimizer.zero_grad()
				sampled_data = sampled_data.to(self.device)
				pred = self.ft_gnn(self.feature_dict, sampled_data.x_dict, sampled_data.y_dict, sampled_data.edge_index_dict, sampled_data['target_entity', 'linked_to', 'aspect'].edge_label_index)
				edge_label = sampled_data['target_entity', 'linked_to', 'aspect'].edge_label
				edge_label = edge_label.to(self.device)
				loss = self.loss_fn(pred, edge_label)
				loss.backward()
				optimizer.step()
				loss_val.append(loss.item())
			end = len(loss_val)
			avg_loss_epoch.append(mean(loss_val[start : end + 1]))
			with torch.no_grad():
				self.ft_gnn.eval()
				trainloss = self.evaluate(data_loader = train_loader, tag = 'train')
				valloss = self.evaluate(data_loader = val_loader, tag = 'val')
				if valloss < best_devloss:
					best_devloss = valloss
					bestepoch = epoch
					torch.save(self.ft_gnn.state_dict(),
			                       f"{CKP}\\{dataset_type}_{content}_{self.task}_{self.mode}.pt")
				if lr_scheduler is not None:
					lr_scheduler.step(valloss)

				train_loss.append(trainloss)
				valid_loss.append(valloss)

			print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, trainloss, valloss))

			if epoch - bestepoch >= 10:
				print("Early stopping")
				break

		print('Epoch: %d / %d, Train loss: %0.6f, Valid loss: %0.6f' % (epoch, epochs, self.evaluate(data_loader = train_loader, tag = 'train'), self.evaluate(data_loader = val_loader, tag = 'val')))

	def get_test_id(self, pred_dict, asp_id_key, target_node_key):
		final_dict = {}
		print('Getting predictions')
		for node_id in target_node_key:
			for asp_id in asp_id_key:
				key = torch.tensor([target_node_key[node_id], asp_id_key[asp_id]])
				if key in pred_dict:
					final_dict[(node_id, asp_id)] = [pred_dict[key], 1]
				else:
					final_dict[(node_id, asp_id)] = [0, 0]

		print(len(final_dict))


	def predict(self, test_loader, root_csv, CSV = CSV):
		self.ft_gnn.eval()
		total_loss = 0	
		preds = []
		ground_truths = []

		self.ft_gnn.eval()
		self.ft_gnn.to(self.device)

		pred_dict = {}

		with torch.no_grad():
			for sampled_data in tqdm.tqdm(test_loader):
				sampled_data = sampled_data.to(self.device)
				target = sampled_data['target_entity', 'aspect'].edge_label
				pred = self.ft_gnn(self.feature_dict, sampled_data.x_dict, sampled_data.y_dict, sampled_data.edge_index_dict, sampled_data['target_entity', 'aspect'].edge_label_index, mode = 'test')
				test_loss = self.loss_fn(pred, target)
				pred_prob = torch.sigmoid(pred)
				preds.append(pred_prob)
				i = 0
				for el in sampled_data['target_entity', 'aspect'].edge_label_index.T:
					pred_dict[el] = pred_prob[i]
					i += 1
				ground_truths.append(target)
				total_loss += test_loss


		pred = torch.cat(preds, dim=0) # Joins the two tensors
		pred = pred.detach()
		ground_truth = torch.cat(ground_truths, dim=0)
		ground_truth = ground_truth.detach()

		y_pred_class = torch.tensor([1 if el > 0.5 else 0 for el in pred]).to(self.device)
		print('Ground Truth', ground_truth.shape)
		utils.write_to_file('ground_truth.txt', ground_truth)
		print('Pred Class', y_pred_class.shape)
		utils.write_to_file('pred.txt', pred)
		f1_sc = multiclass_f1_score(ground_truth, y_pred_class, num_classes = 2)
		avg_loss = total_loss / len(test_loader)
		print('Test loss is %0.6f' % avg_loss)
		print('Test F1 score is %0.6f' % f1_sc)
		return pred_dict

def run_gnn_finetuned(device, feature_dict, lm, CKP, dtr, nsr, mode, train = True, train_graph = None, val_graph = None, test_graph = None, model_file = None, epochs = 5, lr = 0.001, weight_decay = 1e-4, CSV = None, root_csv = None, batch_size = None, dataset_type = 'train-small', content = 'sentence', task = 'linkpred'):
	start = time.time()
	if not train:
		if root_csv is None:
			print('Please provide a root csv file')
			exit()
		mainclass = Main(feature_dict = feature_dict, lm = lm, graph = test_graph, CKP = CKP, device = device, infer = True, model_file = model_file, task = task, mode = mode)
		test_loader = mainclass.prepare_dataloader(test_graph, batch_size = batch_size, tag = 'test', dtr = dtr, nsr = nsr)
		pred_dict = mainclass.predict(test_loader, root_csv, CSV = CSV)
		#mainclass.get_test_id(pred_dict, asp_id_key, target_node_key)
		#print(len(pred_dict.keys()))
	else: 
		mainclass = Main(feature_dict = feature_dict, lm = lm, graph = train_graph, mode = mode, CKP = CKP, device = device, task = task)
		#print(batch_size)
		print(batch_size)
		train_loader = mainclass.prepare_dataloader(train_graph, batch_size = batch_size, dtr = dtr, nsr = nsr)
		sampled_data = next(iter(train_loader))
		#print(sampled_data)
		val_loader = mainclass.prepare_dataloader(val_graph, batch_size = batch_size, tag = 'val', dtr = dtr, nsr = nsr)
		weight_decay = 1e-4
		opt = optim.Adam(mainclass.ft_gnn.parameters(), lr = lr, weight_decay = weight_decay)
		lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 4)
		mainclass.train(train_loader, val_loader, epochs, opt, lr_scheduler)

	end = time.time()
	if train:
		print(f'Total time taken train the Graph Neural Network is {round((end - start), 2)} seconds')

	else:
		print(f'Total time taken test the Graph Neural Network is {round((end - start), 2)} seconds')