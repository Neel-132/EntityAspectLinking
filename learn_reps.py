import embedding
from transformers import BertTokenizer
from tqdm import tqdm
import torch
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def learn_word_emb(data, subject = 'target_entity', pretrained = 'bert-base-uncased', dim = 150):
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	pretrained = pretrained
	ent_emb = embedding.EntityEmbedding(pretrained = pretrained, device = device)
	if subject not in ['target_entity', 'aspect', 't_entitites', 'a_entities']:
		print('Please select one of target_entity, aspect, t_entitites, a_entities as subject'):
		return
	elif subject == 'target_entity':
		size = len(data)
		target_ent_emb = torch.zeros(size, dim)
		target_ent_emb = target_ent_emb.to(device)
		print('Learning target entity embeddings.....')
		for i in tqdm(range(size)):
			entity = data[i]['target_entity']
			if len(entity) == 0:
				continue
			tokens = tokenizer.tokenize(entity)
			input_ids = tokenizer.convert_tokens_to_ids(tokens)
    		input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    		target_emb[i] = ent_emb(input_ids, dim = 150)
    	print('Done learning target entity embeddings.')
    	return target_emb

    elif subject == 't_entitites':
    	t_entity_key = {}
    	size = len(data)
    	print('Learning embeddings of entities associated to context.....')
    	for i in tqdm(range(size)):
    		for ent in data[i]['entities']:
    			eid = ent['entity_id']
    			if eid not in t_entity_key.keys():
    				entity = ent['entity']
    				if len(entity) == 0:
    					continue
    				tokens = tokenizer.tokenize(entity)
    				input_ids = tokenizer.convert_tokens_to_ids(tokens)
    				input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    				t_entity_key[eid] = ent_emb(input_ids, dim)
    	t_entity_emb = torch.zeros(len(t_entity_key.keys()), dim)
    	for el in t_entity_key:
    		t_entity_emb[i] = t_entity_key[el]
    		i += 1
    	print('Done learning context associated entity embeddings.')
    	return t_entity_key, t_entity_emb

    elif subject == 'aspect':
    	pass



