import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero,GATConv, GCNConv
import torch.nn.functional as F
import torch.nn as nn
import torch

class GraphSage(nn.Module):
	""" Class to Implement the GraphSageModule """
	def __init__(self,hidden_channel,out_channel,activation):
		super().__init__()
		self.activation = activation
		self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channel)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
        x = self.conv2(x, edge_index)

        return x

class GraphAttention(nn.Module):
	""" Class to implement Graph Attention Module """

	def __init__(self,hidden_channel,out_channel,activation = "relu",heads = 5):
		super().__init__()
		self.activation = activation
		self.heads = heads
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops = False)
        self.conv2 = GATConv((-1, -1), out_channel, add_self_loops = False)

    def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
        x = self.conv2(x, edge_index)

        return x

class GraphConvolution(nn.Module):
	""" Class to implement Graph Convolution Module """
	def __init__(self, hidden_channel,out_channel,activation = "relu"):
		super().__init__()
		self.activation = activation
		self.conv1 = GCNConv(-1,hidden_channels)
		self.conv2 = GCNConv(-1, out_channel)
		

	def forward(self,x,edge_index):
		x = self.conv1(x, edge_index)
		x = self.activation(x)
        x = self.conv2(x, edge_index)

        return x

class Classifier(nn.Module):
    def __init__(self, hidden_channels,hidden_channels2):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels2)
        self.lin2 = torch.nn.Linear(hidden_channels2, 1)

        def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['entity'][row], z_dict['aspect'][col]], dim=-1) #Change to correct aspect and entity names

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(nn.Module):
	def __init__(self,hidden_channels_gnn,out_channels,graph,hidden_channels_lin1,hidden_channels_lin2,activation = "relu",aggregation = "mean" ,encoder_type = "GAT"):
		super().__init__()
		self.lin = nn.Linear(inp_channel,150)
		self.encoder_type = encoder_type
		if self.encoder_type == "GAT":
			self.encoder = GraphAttention(hidden_channels_gnn,out_channels,activation)
		elif self.encoder_type == "GCN":
			self.encoder = GraphConvolution(hidden_channels_gnn,out_channels,activation)
		elif self.encoder_type == "GSG":
			self.encoder == GraphSage(hidden_channels_gnn,out_channels,activation)
		else:
			pass

		self.encoder = to_hetero(self.encoder, graph.metadata(), aggr=aggregation)

		self.decoder = Classifier(hidden_channels_lin1,hidden_channels_lin2)

	def forward(self, x_dict, y_dict, edge_index_dict, edge_label_index):
		x_dict["target_entity"] = self.lin(torch.cat((x_dict["target_entity"],y_dict["target_entity"]),dim=1))  
        z_dict = self.encoder(x_dict, edge_index_dict)
		output = self.decoder(z_dict, edge_label_index)
		return output


