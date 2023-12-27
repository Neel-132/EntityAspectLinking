import embedding
from transformers import BertTokenizer
from tqdm import tqdm
import torch
import pickle
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]","",text)
    text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))","",text)
    text = re.sub("<(\"[^\"]*\"|'[^']*'|[^'\">])*>","",text)    
    return text

def learn_word_emb(data, subject = 'target_entity', pretrained = 'bert-base-uncased', dim = 768):
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	pretrained = pretrained
	ent_emb = embedding.EntityEmbedding(pretrained = pretrained, device = device)
	if subject not in ['target_entity', 'aspect', 't_entitites', 'a_entities']:
		print('Please select one of target_entity, aspect, t_entitites, a_entities as subject')
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
			target_ent_emb[i] = ent_emb(input_ids, dim = 768)
		print('Done learning target entity embeddings.')
		return target_ent_emb

	elif subject == 't_entitites':
		t_entity_key = {}
		size = len(data)
		ind = 0
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
			t_entity_emb[ind] = t_entity_key[el]
			ind += 1
		print('Done learning context associated entity embeddings.')
		return t_entity_key, t_entity_emb

	elif subject == 'aspect':
		aspect_key = {}
		print('Learning aspect embeddings......')
		for i in tqdm(range(len(data))):
		    aspect = data[i]['true_aspect']
		    true_aspect_id = data[i]['true_aspect_id']
		    tokens = tokenizer.tokenize(aspect)
		    input_ids = tokenizer.convert_tokens_to_ids(tokens)
		    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
		    if true_aspect_id not in aspect_key:
		        aspect_key[true_aspect_id] = ent_emb(input_ids, dim = dim)
		    for element in data[i]['candidate_aspects']:
		        if len(element['aspect_name']) == 0:
		            continue
		        if element['aspect_id'] != true_aspect_id:
		            if element['aspect_id'] not in aspect_key:
		                tokens = tokenizer.tokenize(element['aspect_name'])
		                input_ids = tokenizer.convert_tokens_to_ids(tokens)
		                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
		                aspect_key[element['aspect_id']] = ent_emb(input_ids, dim = dim)

		asp_emb = torch.zeros(len(aspect_key.keys()), dim).to(device)
		ind = 0
		for el in aspect_key:
			asp_emb[ind] = aspect_key[el]
			ind += 1
		print('Done learning aspect embeddings.')
		return aspect_key, asp_emb

	elif subject == 'a_entities':
		a_entity_key = {}
		print('Learning embeddings of entities associated to aspect content.....')
		for i in tqdm(range(len(data))):
		    tasp_id = data[i]['true_aspect_id']
		    for el in data[i]['candidate_aspects']:
		        asp_id = el['aspect_id']
		        for ent in el['entities']:
		            if ent['entity_id'] not in a_entity_key.keys():
		                entity = ent['entity_name']
		                token = tokenizer.tokenize(entity)
		                input_ids = tokenizer.convert_tokens_to_ids(token)
		                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
		                a_entity_key[ent['entity_id']] = ent_emb(input_ids, dim = dim)
		a_entity_emb = torch.zeros(len(a_entity_key), dim)
		ind = 0
		for el in a_entity_key:
			a_entity_emb[ind] = a_entity_key[el]
		print('Done learning aspect content associated entity embeddings.')
		return a_entity_key, a_entity_emb

def learn_text_emb(data, subject = 'context', tag = 'sentence', pretrained = 'all-MiniLM-L6-v2'):
	text_emb = embedding.TextEmbedding(device = device)
	if subject not in ['context', 'aspect content']:
		print('Please select one of context or aspect content as the subject')
		exit()
	if tag not in ['paragraph', 'sentence']:
		print('Please select one of sentence or paragraph as the tag')
		exit()
	elif subject == 'context':
		context_emb = torch.zeros(len(data), 384).to(device)
		print('Learning context embeddings....')
		if tag == 'sentence':
			for i in tqdm(range(len(data))):
				sentence = data[i]['sentence']
				processed_sent = preprocess(sentence)
				output = text_emb(processed_sent)
				context_emb[i] = output
			print('Done learning context embeddings.')
			return context_emb
		elif tag == 'paragraph':
			for i in tqdm(range(len(data))):
				paragraph = data[i]['paragraph']
				tokens = paragraph.split('.')
				token_emb = torch.zeros(len(tokens), 384).to(device)
				for ind in range(len(tokens)):
					preprocessed_token = preprocess(tokens[ind])
					token_output = text_emb(preprocessed_token)
					token_emb[ind] = token_output
				context_emb[i] = torch.mean(token_emb, dim = 0)
			print('Done learning context embeddings.')
			return context_emb
	elif subject == 'aspect content':
		content_emb = torch.zeros(len(data), 384)
		if tag != 'paragraph':
			print('Aspect content tag cannot be other than paragraph')
			return
		else:
			print('Learning representations of aspect content....')
			for i in range(len(data)):
				paragraph = data[i]['content']
				tokens = paragraph.split('.')
				token_emb = torch.zeros(len(tokens), 384)
				for ind in range(len(tokens)):
					preprocessed_token = preprocess(tokens[ind])
					token_output = text_emb(preprocessed_token)
					token_emb[ind] = token_output
				content_emb[i] = torch.mean(token_emb, dim = 0)
			print('Done learning aspect content representations.')
			return content_emb

def learn_bert_text_emb(data, subject = 'context', tag = 'sentence', pretrained = 'bert-base-uncased'):
	tokenizer = BertTokenizer.from_pretrained(pretrained)
	pretrained = pretrained
	text_emb = embedding.EntityEmbedding(pretrained = pretrained, device = device)
	if subject not in ['context', 'aspect content']:
		print('Please select one of context or aspect content as the subject')
		exit()
	if tag not in ['paragraph', 'sentence']:
		print('Please select one of sentence or paragraph as the tag')
		exit()	

	if subject == 'context':
		context_emb = torch.zeros(len(data), 768).to(device)
		if tag == 'sentence':
			for i in tqdm(range(len(data))):
				sentence = data[i]['sentence']
				processed_sent = preprocess(sentence)
				encoded_dict = tokenizer(processed_sent, add_special_tokens = True, max_length = 100, 
					pad_to_max_length = True, return_tensors = 'pt', truncation = True)

				'''processed_sent = '[CLS]' + processed_sent + '[SEP]'
				tokenized_text = tokenizer.tokenize(processed_sent)
				if i == 564:
					print(len(tokenized_text))
				segments_ids = [1] * len(tokenized_text)
				input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
				if i == 564:
					print(len(input_ids))
				input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
				segment_tensors = torch.tensor(segments_ids).unsqueeze(0).to(device)				
				output = text_emb(input_ids, segment_tensors = segment_tensors)'''
				for key in encoded_dict:
					encoded_dict[key] = encoded_dict[key].to(device)
				output = text_emb(encoded_dict = encoded_dict) 
				context_emb[i] = output
			print('Done learning context embeddings.')
			return context_emb

	else:
		pass










