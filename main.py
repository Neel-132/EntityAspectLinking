import utils
import os
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
	parser.add_argument('-tsd', '--testingdataset', type = str, choices = ['test', 'nanni-test', 'nanni-201'], default = 'test', 
		help = 'Testing dataset choices: test, nanni-test, nanni-201')
	parser.add_argument('-p', '--pretrainedmodel', type = str, choices = ['bert_base', 'bert-large', 'roberta'])
	parser.add_argument('-b', '--baseline', action = 'store_true', help = 'Train the baseline models')
	parser.add_argument('-g', '--gnn', action = 'store_true', help = 'Train the gnn models')
	parser.add_argument('-m', '--baselinemodel', choices = ['xgboost', 'svm', 'dnn'],
		help = 'Baseline models choices : XGBoost, Support Vector Machines, Deep Neural Network')
	parser.add_argument('-gnn', '--gnnmodel', choices = ['gcn', 'gsg', 'gat'],
		help = 'Graph Neural Network model choices: GCN, GraphSAGE, GAT')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	config = utils.load_config('config.yaml')
	PKL = config['Main']['pickle']
	RUN = config['Main']['run']
	CSV = config['Main']['csv']
	CKP = config['Main']['checkpoint']
	EAL = config['Main']['data']
	
	

	args = parse_args()

	if args.trainingdataset not in ['train-small', 'train_remaining']:
		print('Please provide a particular trainingdataset')
		exit()
	if args.testingdataset not in ['test', 'nanni-test', 'nanni-201']:
		print('Please provide a particular testingdataset')
		exit()
	if args.content not in ['sentence', 'paragraph']:
		print('Please provide correct content type')
		exit()
	train = args.trainingdataset
	val = args.validationdataset
	test = args.testingdataset
	content = args.content
	train_path = f'{PKL}\\{train}_{content}.pkl'
	path = train_path
	if not os.path.isfile(path):
			train_data = utils.unzip_file(path = EAL, filename = f'{train}.jsonl.gz')
			tlngth = len(train_data.keys())
			train_ent = utils.make_entdict(train_data, tag = content, n = tlngth)
			train_asp = utils.make_aspectdict(train_data, n = tlngth)
			train_eal = utils.get_entasp_pair(train_ent, train_asp)
			utils.save_file(f'{train}_{content}.pkl', train_eal, PKL = PKL)
	else:
		train_eal = utils.read_file(f'{train}_{content}.pkl', PKL = PKL)

	val_path = f'{PKL}\\{val}_{content}.pkl'
	path = val_path
	if not os.path.isfile(path):
		val_data = utils.unzip_file(path = EAL, filename = f'{val}.jsonl.gz')
		vlngth = len(val_data.keys())
		val_ent = utils.make_entdict(val_data, tag = content, n = vlngth)
		val_asp = utils.make_aspectdict(val_data, n = vlngth)
		val_eal = utils.get_entasp_pair(val_ent, val_asp)
		utils.save_file(f'{val}_{content}.pkl', val_eal, PKL = PKL)
	else:
		val_eal = utils.read_file(f'{val}_{content}.pkl', PKL = PKL)

	test_path = f'{PKL}\\{test}_{content}.pkl'
	path = test_path
	if not os.path.isfile(path):
		test_data = utils.unzip_file(path = EAL, filename = f'{test}.jsonl.gz')
		tstlngth = len(test_data.keys())
		test_ent = utils.make_entdict(test_data, tag = content, n = tstlngth)
		test_asp = utils.make_aspectdict(test_data, n = tstlngth)
		test_eal = utils.get_entasp_pair(test_ent, test_asp)
		utils.save_file(f'{test}_{content}.pkl', test_eal, PKL = PKL)
	else:
		test_eal = utils.read_file(f'{test}_{content}.pkl', PKL = PKL)

	





	
