import utils
from argparse import ArgumentParser

def parse_args():
	parser = ArgumentParser(description = 'Process the command line arguments')
	parser.add_argument('-tsk', '--task', choices = ['linkpred', 'nodecls'], default = 'linkpred', 
		help = 'Task choices : Link Prediction, Node Classification')
	parser.add_argument('-c', '--content', choices = ['sentence', 'paragraph'], default = 'sentence')
	parser.add_argument('-trd', '--trainingdataset', type = str, choices = ['train-small', 'train_remaining'], default = 'train-small', 
		help = 'Training dataset choices: train-small, train_remaining')
	parser.add_argument('-vd', '--validationdataset', type = str, choices = ['validation'], required = True, 
		help = 'Validation dataset choice : validate')
	parser.add_argument('tsd', '--testingdataset', type = str, choices = ['test', 'nanni-test', 'nanni-201'], default = 'test', 
		help = 'Testing dataset choices: test, nanni-test, nanni-201')
	parser.add_argument('b', '--baseline', action = 'store_true', help = 'Train the baseline models')
	parser.add_argument('g', '--gnn', action = 'store_true', help = 'Train the gnn models')
	parser.add_argument('m', '--seqmodel', choices = ['xgboost', 'svm', 'dnn'], 
		help = 'Baseline models choices : XGBoost, Support Vector Machines, Deep Neural Network')
	parser.add_argument('gnn', '--gnnmodel', choices = ['gcn', 'gsg', 'gat'], required = True,
		help = 'Graph Neural Network model choices: GCN, GraphSAGE, GAT')
	args = parser.parse_args()
	return args
