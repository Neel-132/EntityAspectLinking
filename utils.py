import tarfile
import gzip
import os
import json
import pickle
import urllib.parse
import re

EAL = r'D:\Entity Aspect Linking\data\entity-aspect-linking-2020\collection'
PTH = r'.\picklefiles'

def unzip_file(path = EAL, filename = 'train-small.jsonl.gz', encoding = 'UTF-8'):
	data = {}
	i = 0
	with gzip.open(f'{path}\\{filename}', 'rt', encoding = encoding) as zipfile:
		for line in zipfile:
			my_object = json.loads(line)
			data[str(i)] = my_object
			i = i + 1
		print('File read successfully')
	zipfile.close()
	return data


def decode_url(s) -> str:
    s = s.replace("enwiki:","")
    return urllib.parse.unquote(s)


def clean_aspect(s) -> str:
    s = decode(s)
    pattern = re.compile(r'.*?/')
    return re.sub(pattern, '', s)


def make_entdict(data, tag = 'sentence', n = 10, israndom = False, ind = []):
    ent_data = []
    if israndom == True:
        it = list(map(int, ind))
    else:
        it = range(n)
    i = 0
    for i in it:
        temp = {}
        ent = data[str(i+1)]
        temp["id"] = ent["id"]
        temp["target_entity"] = decode(ent['context']['target_entity'])
        temp[tag] = ent['context'][tag]['content']
        temp["entities"] = []                        
        k = 0
        for j in ent['context'][tag]['entities']:
            ent = {}                          
            if not j["target_mention"]:
                ent['entity_id'] = j['entity_id']
                ent['entity'] = j['entity_name']
                ent['mention'] = j['mention']
                temp["entities"].append(ent)
            k += 1
        ent_data.append(temp)    
    return ent_data


def get_aspectdict(data, n = 10, israndom = False, ind = []):
    aspects = []
    k = 0
    if israndom == True:
        it = list(map(int, ind))
    else:
        it = range(n)
    for i in it:
        temp = {}
        ent = data[str(i + 1)]
        temp['true_aspect_id'] = ent['true_aspect']
        temp['true_aspect'] = clean_aspect(ent['true_aspect'])
        candasp = []
        j = 0
        for cand in ent['candidate_aspects']:
            casp = {}
            casp['aspect_id'] = cand['aspect_id']
            casp['aspect_name'] = cand['aspect_name']
            casp['section_heading'] = cand['location']['section_headings']
            casp['content'] = cand['aspect_content']['content']
            ent = []
            for e in cand['aspect_content']['entities']:
                temp2 = {}
                temp2['entity_name'] = e['entity_name']
                temp2['entity_id'] = e['entity_id']
                temp2['mention'] = e['mention']
                ent.append(temp2)
                k+= 1
            casp['entities'] = ent
            candasp.append(casp)
            j += 1
        temp['candidate_aspects'] = candasp
        aspects.append(temp)
    return aspects

def get_entasp_pair(ent_dict, asp_dict):
	assert len(ent_dict) != len(asp_dict), 'Entity dict and Aspect dict must have the same length.'
	final_data = [(ent_dict[i], asp_dict[i]) for i in range(len(ent_dict))]
	return final_data


def get_baselinedataset(data, entity_emb, context_emb, aspect_dict):
    data_key = {}
    j = 0
    for i in range(len(data)):
        tup = ()
        ent, asp = data[i]
        ent_emb = entity_emb[i]
        context = context_emb[i]
        true_asp_id = asp['true_aspect_id']
        asp_emb = asp_dict[asp_id]
        tup += (ent_emb, context, asp_emb, 1)
        data_key[j] = tup
        j += 1
        for item in asp['candidate_aspects']:
            tup = ()
            if len(item['aspect_name']) == 0:
                continue
            if item['aspect_id'] != true_asp_id:
                tup += (ent_emb, context, asp_dict[item['aspect_id']], 0)
                data_key[j] = tup
                j += 1
    


def save_file(output_filename, data, path = PTH) -> None:
	with open(f'{PTH}\\{output_filename}', 'wb') as f:
		pickle.dump(data, f)
	print('File dumped successfully!')
	f.close()
	return

def read_file(input_filename, path = PTH) -> dict[int]:
    with open(f'{PTH}\\{input_filename}', 'rb') as f:
        data = pickle.load(f)
    print('File read successfully')
    f.close()
    return data








