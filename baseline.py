from dnn import run_dnn
import ml

def run_baseline(train_data, val_data, test_data, train = True, model = 'xgboost', **kwargs):
	if model not in ['xgboost', 'svm', 'dnn']:
		print('Please select one of xgboost, svm or dnn')
		return
	elif model == 'xgboost':
		pass

	elif model == 'svm':
		pass

	elif model == 'dnn':
		if train != True:
			print('Testing the performance of Deep Neural Network')
			run_dnn(train = False, device = device, test_data = test_data, model_file = model_file, CKP = CKP)
		else:
			print('Intilializing training of Deep Neural Network')
			run_dnn(train_data = train_data, val_data = val_data, batch_size = batch_size, epochs = epochs, lr = learning_rate, CKP = CKP, device = device, weight_decay = weight_decay)

