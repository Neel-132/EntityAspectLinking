import utils
import os
from argparse import ArgumentParser
import learn_reps
import torch
import dnn

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
	parser.add_argument('-p', '--pretrainedmodel', type = str, choices = ['bert-base', 'bert-large', 'roberta'], default = 'bert-base',
		help  = 'Pretrained model choices: BERT base, BERT large, RoBERTa')
	parser.add_argument('-b', '--baseline', action = 'store_true', help = 'Train the baseline models')
	parser.add_argument('-g', '--gnn', action = 'store_true', help = 'Train the gnn models')
	parser.add_argument('-m', '--baselinemodel', choices = ['xgboost', 'svm', 'dnn'],
		help = 'Baseline models choices : XGBoost, Support Vector Machine, Deep Neural Network')
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
	epochs = config['DNN']['Train']['epochs']
	lr = config['DNN']['Train']['lr']
	weight_decay = config['DNN']['Train']['weight_decay']
	train_batch_size = config['DNN']['Train']['batch_size']
	train_ratio = config['DNN']['Train']['train_ratio']
	val_ratio = config['DNN']['Train']['val_ratio']
	test_batch_size = config['DNN']['Test']['batch_size']


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
	if args.pretrainedmodel not in ['bert-base', 'bert-large', 'roberta']:
		print('Please provide correct pretrained model type')
		exit()
	if args.baselinemodel not in ['dnn', 'xgboost', 'svm']:
		print('Please provide correct baseline model')

	train = args.trainingdataset
	val = args.validationdataset
	test = args.testingdataset
	content = args.content
	train_path = f'{PKL}\\{train}_{content}.pkl'
	path = train_path
	pretrainedmodel = args.pretrainedmodel
	baselinemodel = args.baselinemodel
	if pretrainedmodel == 'roberta':
		pass 
	else:
		pretrained = pretrainedmodel+'-uncased'

	# Preparing the (entity,aspect) tuples

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

	# Learning representations using pretrained BERT/RoBERTa

	train_entembpath = f'{PKL}\\{train}_{content}_{pretrainedmodel}_targetentemb.pkl'
	path = train_entembpath

	train_ent = [item[0] for item in train_eal]
	train_asp = [item[1] for item in train_eal]

	if not os.path.isfile(path):
		train_target_ent_emb = learn_reps.learn_word_emb(train_ent, pretrained = pretrained)
		utils.save_file(f'{train}_{content}_{pretrainedmodel}_targetentemb.pkl', train_target_ent_emb.cpu(), PKL = PKL)

	else:
		train_target_ent_emb = utils.read_file(f'{train}_{content}_{pretrainedmodel}_targetentemb.pkl', PKL = PKL)

	train_aspembpath = f'{PKL}\\{train}_{content}_{pretrainedmodel}_aspemb.pkl'
	path = train_aspembpath

	if not os.path.isfile(path):
		train_asp_key, train_asp_emb = learn_reps.learn_word_emb(train_asp, pretrained = pretrained, subject = 'aspect')
		utils.save_file(f'{train}_{content}_{pretrainedmodel}_aspemb.pkl', train_asp_emb.cpu(), PKL = PKL)
		
		for key in train_asp_key:
			train_asp_key[key] = train_asp_key[key].cpu()

		utils.save_file(f'{train}_{content}_{pretrainedmodel}_aspdict.pkl', train_asp_key, PKL = PKL)
	else:
		train_asp_emb = utils.read_file(f'{train}_{content}_{pretrainedmodel}_aspemb.pkl', PKL = PKL)
		train_asp_key = utils.read_file(f'{train}_{content}_{pretrainedmodel}_aspdict.pkl', PKL = PKL)

	train_context_path = f'{PKL}\\{train}_{content}_contextemb.pkl'
	path = train_context_path

	if not os.path.isfile(path):
		train_context_emb = learn_reps.learn_text_emb(train_ent, tag = content)
		utils.save_file(f'{train}_{content}_contextemb.pkl', train_context_emb.cpu(), PKL = PKL)
	else:
		train_context_emb = utils.read_file(f'{train}_{content}_contextemb.pkl', PKL = PKL)

	train_t_entities_path = f'{PKL}\\{train}_{content}_{pretrainedmodel}_t_entities_emb.pkl'
	path = train_t_entities_path

	if not os.path.isfile(path):
		train_t_entity_key, train_t_entity_emb = learn_reps.learn_word_emb(train_ent, pretrained = pretrained, subject = 't_entitites')
		utils.save_file(f'{train}_{content}_{pretrainedmodel}_t_entities_emb.pkl', train_t_entity_emb.cpu(), PKL = PKL)
		
		for key in train_t_entity_key:
			train_t_entity_key[key] = train_t_entity_key[key].cpu()

		utils.save_file(f'{train}_{content}_{pretrainedmodel}_t_entitydict.pkl', train_t_entity_key, PKL = PKL)

	else:
		train_t_entity_emb = utils.read_file(f'{train}_{content}_{pretrainedmodel}_t_entities_emb.pkl', PKL = PKL)
		train_t_entity_key = utils.read_file(f'{train}_{content}_{pretrainedmodel}_t_entitydict.pkl', PKL = PKL)

	train_a_entities_path = f'{PKL}\\{train}_{content}_{pretrainedmodel}_a_entities_emb.pkl'
	path = train_a_entities_path

	if not os.path.isfile(path):
		train_a_entity_key, train_a_entity_emb = learn_reps.learn_word_emb(train_asp, pretrained = pretrained, subject = 'a_entities')
		utils.save_file(f'{train}_{content}_{pretrainedmodel}_a_entities_emb.pkl', train_a_entity_emb.cpu(), PKL = PKL)
		
		for key in train_a_entity_key:
			train_a_entity_key[key] = train_a_entity_key[key].cpu()

		utils.save_file(f'{train}_{content}_{pretrainedmodel}_a_entitydict.pkl', train_a_entity_key, PKL = PKL)

	else:
		train_a_entity_emb = utils.read_file(f'{train}_{content}_{pretrainedmodel}_a_entities_emb.pkl', PKL = PKL)
		train_a_entity_key = utils.read_file(f'{train}_{content}_{pretrainedmodel}_a_entitydict.pkl', PKL = PKL)

	val_entembpath = f'{PKL}\\{val}_{content}_{pretrainedmodel}_targetentemb.pkl'
	path = val_entembpath

	val_ent = [item[0] for item in val_eal]
	val_asp = [item[1] for item in val_eal]

	if not os.path.isfile(path):
		val_target_ent_emb = learn_reps.learn_word_emb(val_ent, pretrained = pretrained)
		utils.save_file(f'{val}_{content}_{pretrainedmodel}_targetentemb.pkl', val_target_ent_emb.cpu(), PKL = PKL)

	else:
		val_target_ent_emb = utils.read_file(f'{val}_{content}_{pretrainedmodel}_targetentemb.pkl', PKL = PKL)

	val_aspembpath = f'{PKL}\\{val}_{content}_{pretrainedmodel}_aspemb.pkl'
	path = val_aspembpath

	if not os.path.isfile(path):
		val_asp_key, val_asp_emb = learn_reps.learn_word_emb(val_asp, pretrained = pretrained, subject = 'aspect')
		utils.save_file(f'{val}_{content}_{pretrainedmodel}_aspemb.pkl', val_asp_emb.cpu(), PKL = PKL)
		
		for key in val_asp_key:
			val_asp_key[key] = val_asp_key[key].cpu()

		utils.save_file(f'{val}_{content}_{pretrainedmodel}_aspdict.pkl', val_asp_key, PKL = PKL)
	else:
		val_asp_emb = utils.read_file(f'{val}_{content}_{pretrainedmodel}_aspemb.pkl', PKL = PKL)
		val_asp_key = utils.read_file(f'{val}_{content}_{pretrainedmodel}_aspdict.pkl', PKL = PKL)

	val_context_path = f'{PKL}\\{val}_{content}_contextemb.pkl'
	path = val_context_path

	if not os.path.isfile(path):
		val_context_emb = learn_reps.learn_text_emb(val_ent, tag = content)
		utils.save_file(f'{val}_{content}_contextemb.pkl', val_context_emb.cpu(), PKL = PKL)
	else:
		val_context_emb = utils.read_file(f'{val}_{content}_contextemb.pkl', PKL = PKL)

	val_t_entities_path = f'{PKL}\\{val}_{content}_{pretrainedmodel}_t_entities_emb.pkl'
	path = val_t_entities_path

	if not os.path.isfile(path):
		val_t_entity_key, val_t_entity_emb = learn_reps.learn_word_emb(val_ent, pretrained = pretrained, subject = 't_entitites')
		utils.save_file(f'{val}_{content}_{pretrainedmodel}_t_entities_emb.pkl', val_t_entity_emb.cpu(), PKL = PKL)
		
		for key in val_t_entity_key:
			val_t_entity_key[key] = val_t_entity_key[key].cpu()

		utils.save_file(f'{val}_{content}_{pretrainedmodel}_t_entitydict.pkl', val_t_entity_key, PKL = PKL)

	else:
		val_t_entity_emb = utils.read_file(f'{val}_{content}_{pretrainedmodel}_t_entities_emb.pkl', PKL = PKL)
		val_t_entity_key = utils.read_file(f'{val}_{content}_{pretrainedmodel}_t_entitydict.pkl', PKL = PKL)

	val_a_entities_path = f'{PKL}\\{val}_{content}_{pretrainedmodel}_a_entities_emb.pkl'
	path = val_a_entities_path

	if not os.path.isfile(path):
		val_a_entity_key, val_a_entity_emb = learn_reps.learn_word_emb(val_asp, pretrained = pretrained, subject = 'a_entities')
		utils.save_file(f'{val}_{content}_{pretrainedmodel}_a_entities_emb.pkl', val_a_entity_emb.cpu(), PKL = PKL)
		
		for key in val_a_entity_key:
			val_a_entity_key[key] = val_a_entity_key[key].cpu()

		utils.save_file(f'{val}_{content}_{pretrainedmodel}_a_entitydict.pkl', val_a_entity_key, PKL = PKL)

	else:
		val_a_entity_emb = utils.read_file(f'{val}_{content}_{pretrainedmodel}_a_entities_emb.pkl', PKL = PKL)
		val_a_entity_key = utils.read_file(f'{val}_{content}_{pretrainedmodel}_a_entitydict.pkl', PKL = PKL)


	test_entembpath = f'{PKL}\\{test}_{content}_{pretrainedmodel}_targetentemb.pkl'
	path = test_entembpath

	test_ent = [item[0] for item in test_eal]
	test_asp = [item[1] for item in test_eal]

	if not os.path.isfile(path):
		test_target_ent_emb = learn_reps.learn_word_emb(test_ent, pretrained = pretrained)
		utils.save_file(f'{test}_{content}_{pretrainedmodel}_targetentemb.pkl', test_target_ent_emb.cpu(), PKL = PKL)

	else:
		test_target_ent_emb = utils.read_file(f'{test}_{content}_{pretrainedmodel}_targetentemb.pkl', PKL = PKL)

	test_aspembpath = f'{PKL}\\{test}_{content}_{pretrainedmodel}_aspemb.pkl'
	path = test_aspembpath

	if not os.path.isfile(path):
		test_asp_key, test_asp_emb = learn_reps.learn_word_emb(test_asp, pretrained = pretrained, subject = 'aspect')
		utils.save_file(f'{test}_{content}_{pretrainedmodel}_aspemb.pkl', test_asp_emb.cpu(), PKL = PKL)
		for key in test_asp_key:
			test_asp_key[key] = test_asp_key[key].cpu()

		utils.save_file(f'{test}_{content}_{pretrainedmodel}_aspdict.pkl', test_asp_key, PKL = PKL)
	else:
		test_asp_emb = utils.read_file(f'{test}_{content}_{pretrainedmodel}_aspemb.pkl', PKL = PKL)
		test_asp_key = utils.read_file(f'{test}_{content}_{pretrainedmodel}_aspdict.pkl', PKL = PKL)

	test_context_path = f'{PKL}\\{test}_{content}_contextemb.pkl'
	path = test_context_path

	if not os.path.isfile(path):
		test_context_emb = learn_reps.learn_text_emb(test_ent, tag = content)
		utils.save_file(f'{test}_{content}_contextemb.pkl', test_context_emb.cpu(), PKL = PKL)
	else:
		test_context_emb = utils.read_file(f'{test}_{content}_contextemb.pkl', PKL = PKL)

	test_t_entities_path = f'{PKL}\\{test}_{content}_{pretrainedmodel}_t_entities_emb.pkl'
	path = test_t_entities_path

	if not os.path.isfile(path):
		test_t_entity_key, test_t_entity_emb = learn_reps.learn_word_emb(test_ent, pretrained = pretrained, subject = 't_entitites')
		utils.save_file(f'{test}_{content}_{pretrainedmodel}_t_entities_emb.pkl', test_t_entity_emb.cpu(), PKL = PKL)
		
		for key in test_t_entity_key:
			test_t_entity_key[key] = test_t_entity_key[key].cpu()

		utils.save_file(f'{test}_{content}_{pretrainedmodel}_t_entitydict.pkl', test_t_entity_key, PKL = PKL)

	else:
		test_t_entity_emb = utils.read_file(f'{test}_{content}_{pretrainedmodel}_t_entities_emb.pkl', PKL = PKL)
		test_t_entity_key = utils.read_file(f'{test}_{content}_{pretrainedmodel}_t_entitydict.pkl', PKL = PKL)

	test_a_entities_path = f'{PKL}\\{test}_{content}_{pretrainedmodel}_a_entities_emb.pkl'
	path = test_a_entities_path

	if not os.path.isfile(path):
		test_a_entity_key, test_a_entity_emb = learn_reps.learn_word_emb(test_asp, pretrained = pretrained, subject = 'a_entities')
		utils.save_file(f'{test}_{content}_{pretrainedmodel}_a_entities_emb.pkl',test_a_entity_emb.cpu(), PKL = PKL)
		
		for key in test_a_entity_key:
			test_a_entity_key[key] = test_a_entity_key[key].cpu()

		utils.save_file(f'{test}_{content}_{pretrainedmodel}_a_entitydict.pkl', test_a_entity_key, PKL = PKL)

	else:
		test_a_entity_emb = utils.read_file(f'{test}_{content}_{pretrainedmodel}_a_entities_emb.pkl', PKL = PKL)
		test_a_entity_key = utils.read_file(f'{test}_{content}_{pretrainedmodel}_a_entitydict.pkl', PKL = PKL)

	# Training the baseline models

	if args.baseline:
		train_baselinedataset_path = f'{PKL}\\{train}_{content}_baselinedataset.pkl'
		path = train_baselinedataset_path

		if not os.path.isfile(path):
			train_baselinedataset = utils.get_baselinedataset(train_eal, train_target_ent_emb, train_context_emb, train_asp_key)
			utils.save_file(f'{train}_{content}_baselinedataset.pkl', train_baselinedataset.cpu(), PKL = PKL)
		else:
			train_baselinedataset = utils.read_file(f'{train}_{content}_baselinedataset.pkl', PKL = PKL)

		val_baselinedataset_path = f'{PKL}\\{val}_{content}_baselinedataset.pkl'
		path = val_baselinedataset_path

		if not os.path.isfile(path):
			val_baselinedataset = utils.get_baselinedataset(val_eal, val_target_ent_emb, val_context_emb, val_asp_key)
			utils.save_file(f'{val}_{content}_baselinedataset.pkl', val_baselinedataset.cpu(), PKL = PKL)
		else:
			val_baselinedataset = utils.read_file(f'{val}_{content}_baselinedataset.pkl', PKL = PKL)


		test_baselinedataset_path = f'{PKL}\\{test}_{content}_baselinedataset.pkl'
		path = test_baselinedataset_path

		if not os.path.isfile(path):
			test_baselinedataset = utils.get_baselinedataset(test_eal, test_target_ent_emb, test_context_emb, test_asp_key, train = False, tag = test)
			utils.save_file(f'{test}_{content}_baselinedataset.pkl', test_baselinedataset.cpu(), PKL = PKL)
		else:
			test_baselinedataset = utils.read_file(f'{test}_{content}_baselinedataset.pkl', PKL = PKL)

		dnn_model_file = f'{CKP}\\{train}_{content}_dnn.pt'
		path = dnn_model_file

		if not os.path.isfile(path):
			dnn.run_dnn(device = device, train_data = train_baselinedataset, val_data = val_baselinedataset, epochs = epochs, lr = lr, batch_size = train_batch_size, 
				dataset_type = train, content = content, CKP = CKP, train_ratio = train_ratio, val_ratio = val_ratio)
		else:
			prediction_path = f'{CSV}\\{test}_{content}_{baselinemodel}_pred.csv'
			path = prediction_path
			if not os.path.isfile(path):
				root_csv = f'{test}.csv'
				dnn.run_dnn(device = device, test_data = test_baselinedataset, root_csv = root_csv, CSV = CSV, 
					batch_size = test_batch_size, CKP = CKP, content = content, dataset_type = test, train = False, model_file = f'{train}_{content}_dnn')

			utils.get_run_file(f'{test}_{content}_dnn_pred.csv', content = content, tag = test)
			





















	
