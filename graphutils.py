from torch_geometric.data import HeteroData
import torch
import pickle
import numpy as np
import torch

def get_node_id(data, subject = 'target_entity'):
    res = []
    dic = {}
    if subject == 'target_entity':
        return list(np.arange(len(data)))
    
    elif subject == 't_entities':
        j = 0
        ent = [data[i][0] for i in range(len(data))]
        for el in ent:
            for e in el['entities']:
                if e['original_id'] not in dic:
                    dic[e['original_id']] = j
                    j += 1
        return dic
    
    elif subject == 'aspect_entity':
        asp = [data[i][1] for i in range(len(data))]
        j = 0
        for el in asp:
            tasp_id = el['true_aspect_id']
            if tasp_id not in dic:
                dic[tasp_id] = j
                j += 1
            for cand in el['candidate_aspects']:
                asp_id = cand['aspect_id']
                if asp_id != tasp_id:
                    if asp_id not in dic:
                        dic[asp_id] = j
                        j += 1
        return dic
    
    elif subject == 'a_entities':
        asp = [data[i][1] for i in range(len(data))]
        j = 0
        for item in asp:
            for el in item['candidate_aspects']:
                for ent in el['entities']:
                    if ent['original_id'] not in dic:
                        dic[ent['original_id']] = j
                        j += 1
        return dic

def create_targetedge_index(data, asp_id):
    edge = []
    for i in range(len(data)):
        _, asp = data[i]
        tasp_id = asp['true_aspect_id']
        edge.append((i, asp_id[tasp_id]))
    return torch.tensor(edge).T

def create_associated_edge_index(data, subject = 'target_entity', **kwargs):
    edge = []
    if subject == 'target_entity':
        for i in range(len(data)):
            ent, _  = data[i]
            for item in ent['entities']:
                edge.append((i, associated_entity_dict[item['original_id']]))
        edge = list(set(edge))
        return torch.tensor(edge).T
    
    elif subject == 'aspect_entity':
        for i in range(len(data)):
            _, asp = data[i]
            for element in asp['candidate_aspects']:
                first = asp_dict[element['aspect_id']]
                for ent in element['entities']:
                    edge.append((first, ent_asp_id_dict[ent['original_id']]))
        return torch.tensor(edge).T

