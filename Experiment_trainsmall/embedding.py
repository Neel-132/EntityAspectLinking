import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer

class TextEmbedding(nn.Module):
	def __init__(self, pretrained = "all-MiniLM-L6-v2",device=None):
		super(TextEmbedding, self).__init__()
		#self.pretrained = pretrained
		#self.config = AutoConfig.from_pretrained(self.pretrained)
		#self.bert = AutoModel.from_pretrained(self.pretrained, config = self.config)
		self.device = device
		self.model = SentenceTransformer(pretrained, device=device).to(device)

	@torch.no_grad()
	def forward(self, sent):
		x = self.model.encode(sent , show_progress_bar=True, convert_to_tensor=True, device=self.device)
		return x


class EntityEmbedding(nn.Module):
	def __init__(self, pretrained, device = None):
		super(EntityEmbedding, self).__init__()
		self.device = device
		self.pretrained = pretrained
		self.config = AutoConfig.from_pretrained(self.pretrained)
		self.bert = AutoModel.from_pretrained(self.pretrained, config = self.pretrained)
		self.bert.to(device)

	def forward(self, input_ids, dim = 100):
		with torch.no_grad():
			output = self.bert(input_ids)
			return output[0][:, 0, :dim]



