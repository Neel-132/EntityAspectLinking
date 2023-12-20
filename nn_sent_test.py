import torch
import pickle
from fit_nn_sent_trainsmall import Network
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
CKP = r'./checkpoint'
CSV = r'./csvfiles'
PKL = r'./picklefiles'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def read_pickle(file):
	with open(f'{PKL}//{file}', 'rb') as f:
		data = pickle.load(f)
	f.close()
	return data

def prepare_dataset(file):
	test_data = read_pickle(file)
	features = test_data[:, : -1]
	sc = StandardScaler()
	features = sc.fit_transform(features)
	target = test_data[:, -1]
	return torch.tensor(features, dtype = torch.float32), torch.tensor(target)
def read_model(file):
	return torch.load(f'{CKP}//{file}')
def predict(model_name, file):
	test_data, target = prepare_dataset(file)
	test_data = test_data.to(device)
	target = target.to(device)
	model = Network(584, device = device)
	model.load_state_dict(read_model(model_name))
	model.eval()

	with torch.no_grad():
		y_pred = model(test_data)
		y_pred_class = torch.tensor([1 if x > 0.5 else 0 for x in y_pred]).to(device)
		f1score = multiclass_f1_score(y_pred_class, target)
		print("Test F1-score is %.6f" % (f1score.item()))
		return y_pred_class, y_pred

def get_test_id(test_file, pred_class, pred_prob):	
	test_id = pd.read_csv(f'{CSV}\\{test_file}')
	test_id['Predicted Label'] = pred_class
	test_id['Predicted Probabilities'] = pred_prob
	test_id.to_csv(f'{CSV}\\Predictions_trainsmall_DNN.csv', index = False)
	return

pred, prob = predict('DNN_EAL_trainsmall_para.pt', 'baselinedataset_test.pkl')
#get_test_id('Test_ID.csv', pred.cpu(), prob.cpu())

