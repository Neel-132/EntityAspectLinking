from torch_geometric.data import HeteroData
import torch
import pickle
import numpy as np
import torch
import torch_geometric.transforms as T

def get_node_id1(data, subject = 'target_entity'):
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

def create_node_id(data, subject = 'target_entity'):
    print('Creating node ids...')
    if subject == 'target_entity':
        print('Successfully created target node ids')
        return np.arange(len(data))
    
    elif subject in ['aspect', 't_entities', 'a_entities']:
        key = {}
        i = 0
        for element in data:
            key[element] = i
            i += 1

        print(f'Successfully created {subject} node ids')
        return np.arange(len(data.keys())), key

    else:
        print('Please provide an appropriate subject name')
        exit()


def create_edge_index(data, head, tail, edge_type = 'target'):
    edge = []
    if edge_type not in ['target', 'associated to t_entities', 'associated to a_entities']:
        print('Please provide appropriate edge_type')
        exit()
    elif edge_type == 'target':
        print('Creating target edge mappings...')
        for i in range(len(data)):
            _, aspect = data[i]
            true_asp_id = aspect['true_aspect_id']
            edge.append((head[i], tail[true_asp_id]))
        print('Successfully created target edge mappings.')
        return torch.tensor(edge).T

    elif edge_type == 'associated to t_entities':
        print('Creating associated to target entity edge mappings...')
        for i in range(len(data)):
            entity, _ = data[i]
            for item in entity['entities']:
                ent_id = item['entity_id']
                edge.append((head[i], tail[ent_id]))
        edge = list(set(edge))
        print('Successfully created associated to target entity edge mappings.')
        return torch.tensor(edge).T

    elif edge_type == 'associated to a_entities':
        print('Creating associated to aspect entity edge mappings...')
        for i in range(len(data)):
            _, aspect = data[i]
            for item in aspect['candidate_aspects']:
                if len(item['aspect_name']) == 0:
                    continue
                head_id = head[item['aspect_id']]
                for cand in item['entities']:
                    tail_id = tail[cand['entity_id']]
                    edge.append((head_id, tail_id))
        edge = list(set(edge))
        print('Successfully created associated to aspect entity edge mappings.')
        return torch.tensor(edge).T

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
                if len(element['aspect_name']) == 0:
                    continue
                first = asp_dict[element['aspect_id']]
                for ent in element['entities']:
                    edge.append((first, ent_asp_id_dict[ent['original_id']]))

        return torch.tensor(edge).T

def create_graph(data, entity_emb, context_emb, asp_emb, asp_key, t_entities_emb, t_entities_key, a_entities_emb, a_entities_key, task = 'linkprediction'):

    if task not in ['linkprediction', 'nodeclassification']:
        print('Please provide appropriate task.')
        exit()
    print(f'Creating graph for {task}....')
    if task == 'linkprediction':
        graph = HeteroData()
        target_entity_id = create_node_id(data)
        graph['target_entity'].num_nodes = len(target_entity_id)
        graph['target_entity'].node_id = target_entity_id
        graph['target_entity'].x = entity_emb
        graph['target_entity'].y = context_emb

        aspect_id, asp_id_key = create_node_id(asp_key, subject = 'aspect')
        graph['aspect'].num_nodes = len(aspect_id)
        graph['aspect'].node_id = aspect_id
        graph['aspect'].x = asp_emb

        t_entities_id, t_entities_id_key = create_node_id(t_entities_key, subject = 't_entities')
        graph['t_entities'].num_nodes = len(t_entities_id)
        graph['t_entities'].node_id = t_entities_id
        graph['t_entities'].x = t_entities_emb

        a_entities_id, a_entities_id_key = create_node_id(a_entities_key, subject = 'a_entities')
        graph['a_entities'].num_nodes = len(a_entities_id)
        graph['a_entities'].node_id = a_entities_id
        graph['a_entities'].x = a_entities_emb

        graph['target_entity', 'linked_to', 'aspect'].edge_index= create_edge_index(data, target_entity_id, asp_id_key)
        graph['target_entity', 'associated_to', 't_entities'].edge_index = create_edge_index(data, target_entity_id, t_entities_id_key, edge_type = 'associated to t_entities')
        graph['aspect', 'associated_to', 'a_entities'].edge_index = create_edge_index(data, asp_id_key, a_entities_id_key, edge_type = 'associated to a_entities')

        return T.ToUndirected()(graph)





