import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class TextEmbedding(nn.Module):
	def __init__(self, pretrained):
		super(TextEmbedding, self).__init__()
		self.pretrained = pretrained
		self.config = AutoConfig.from_pretrained(self.pretrained)
		self.bert = AutoModel.from_pretrained(self.pretrained, config = self.config)

	def forward(self, input_ids : torch.tensor, attention_mask : torch.tensor = None):
		output = self.bert(input_ids, attention_mask)
		return output[0][:, 0, :]


class EntityEmbedding(nn.Module):
	def __init__(self, pretrained):
		super(EntityEmbedding, self).__init__()
		self.pretrained = pretrained
		self.config = AutoConfig.from_pretrained(self.pretrained)
		self.bert = AutoModel.from_pretrained(self.pretrained, config = self.pretrained)

	def forward(self, input_ids, dim = 768):
		with torch.no_grad():
			output = self.bert(input_ids)
			return output[0][:, 0, :dim]


