import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer

class TextEmbedding(nn.Module):
	def __init__(self, pretrained = "all-MiniLM-L6-v2",device=None):
		super(TextEmbedding, self).__init__()
		self.device = device
		self.model = SentenceTransformer(pretrained, device=device).to(device)

	@torch.no_grad()
	def forward(self, sent):
		x = self.model.encode(sent , show_progress_bar=False, convert_to_tensor=True, device=self.device)
		return x


class EntityEmbedding(nn.Module):
	def __init__(self, pretrained = "bert-base-uncased", device = None):
		super(EntityEmbedding, self).__init__()
		self.device = device
		self.pretrained = pretrained
		self.config = AutoConfig.from_pretrained(self.pretrained)
		self.bert = AutoModel.from_pretrained(self.pretrained, config = self.pretrained)
		self.bert.to(device)

	def forward(self, input_ids = None, dim = 768, segment_tensors = None, encoded_dict = None):
		with torch.no_grad():
			if encoded_dict is not None:
				output = self.bert(**encoded_dict) #to handle sentence embeddings where input_id length exceeds 512
			else:
				if segment_tensors is not None:
					output = self.bert(input_ids, segment_tensors)	
				else:
					output = self.bert(input_ids)	
			return output.last_hidden_state[:, 0, : dim]






