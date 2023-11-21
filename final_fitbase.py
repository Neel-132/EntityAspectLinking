# %%
#importing the libraries
import torch
import pickle
import numpy as np

#Loading the data
with open(r"C:\Users\SOHAM\Desktop\Entity_Aspect_Linking\EntityAspectLinking\picklefiles\final_eal_random.pkl", 'rb') as eal:
    data = pickle.load(eal)
ent = [data[i][0] for i in range(len(data))]
asp = [data[i][1] for i in range(len(data))]

#Path of the file location
PTH = r"C:\Users\SOHAM\Desktop\Entity_Aspect_Linking\EntityAspectLinking\picklefiles"

# Function to read the embeddings
def read_tensor(filename):
    with open(f'{PTH}\\{filename}', 'rb') as f:
        emb = np.load(f, allow_pickle = True)
    return emb

#Obtaining the embeddings names
ent_emb = read_tensor('random_targetentemb.pkl')
asp_emb = read_tensor('random_aspemb.pkl')
context_emb = read_tensor('random_paragraphemb.pkl')


aspect = [el[1] for el in data]
lngth = [len(el['candidate_aspects']) - 1 for el in aspect]
asp_count = sum([len(el['candidate_aspects']) for el in aspect])

# Function to decode the true and candidate aspect names
import urllib.parse
def decode(str):
    return urllib.parse.unquote(str)

#Case where true aspect is not in the candidate aspect as it is
i = 0
j = 0
check = 0
yo = 0
unid_ls = []
while(i<asp_count):
    aspect = asp[j]["true_aspect"]
    aspect = decode(aspect)
    i = i+1
    this = 0
    for cand in asp[j]["candidate_aspects"]:
        if i>=asp_count:
            break
        if cand['aspect_name'] == aspect:
            check += 1
        if cand['aspect_name'] != aspect:
            this += 1
            casp = cand['aspect_name']
            i = i+1
        if this == len(asp[j]["candidate_aspects"]):
            yo = yo+1
            unid_ls.append(j)
            
#Case where true aspect is not in the candidate aspect as it is and i is stored in the list
i = 0
j = 0
ind = []
while(i<asp_count):
    aspect = asp[j]['true_aspect']
    aspect = decode(aspect)
    i = i+1
    this = 0
    for cand in asp[j]["candidate_aspects"]:
        if i >= asp_count:
            break
        if j in unid_ls and decode(aspect) in str(cand["aspect_name"]):
            ind.append(i)
            i = i+1
            continue
        i = i+1
    j = j+1
    
#Building the dataset for entity and asepct and context
dataset = np.zeros((asp_count, 201 + context_emb.shape[1]))
i = 0
k = 0
for _ in range(len(dataset)):
    while(k < len(lngth)):
        dataset[i, : ent_emb.shape[1]] = ent_emb[k]
        dataset[i, ent_emb.shape[1] : ent_emb.shape[1] + context_emb.shape[1]] = context_emb[k]
        dataset[i, ent_emb.shape[1] + context_emb.shape[1] : -1] = asp_emb[i]
        dataset[i, -1] = 1
        i += 1
        for _ in range(lngth[k]):
            if i in ind:
                print('Here')
                i += 1
                continue
            dataset[i, : ent_emb.shape[1]] = ent_emb[k]
            dataset[i, ent_emb.shape[1] : ent_emb.shape[1] + context_emb.shape[1]] = context_emb[k]
            dataset[i, ent_emb.shape[1] + context_emb.shape[1] : -1] = asp_emb[i]
            dataset[i, -1] = 0
            i += 1
        k += 1
        
#Undersampling to meet class imbalances for class 0 and 1
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
X = dataset[:,:-1]
sc = StandardScaler()
X = sc.fit_transform(X)
y = dataset[:,-1]
undersample = RandomUnderSampler(sampling_strategy='majority')
X, y = undersample.fit_resample(X, y)

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)

#Implementing Xg boost
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  
    'eval_metric': 'logloss',        
    'max_depth': 3,                  
    'learning_rate': 0.1,            
    'subsample': 0.8,                
    'colsample_bytree': 0.8,         
    'seed': 42                       
}
num_rounds = 100  
model = xgb.train(params, dtrain, num_rounds)
y_pred = model.predict(dtest)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]  # Convert probabilities to binary predictions

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

#Implementing Light GBM
import lightgbm as lgb
from sklearn.metrics import  roc_auc_score

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',  # for binary classification 
    'metric': 'auc', # area under the curve
    'boosting_type': 'gbdt',    # traditional Gradient Boosting Decision Tree
    'num_leaves': 31,           # number of leaves in one tree
    'learning_rate': 0.05,      # learning rate
    'feature_fraction': 0.9,    # feature fraction
    'bagging_fraction': 0.8,    # bagging fraction
    'bagging_freq': 5,          # bagging frequency
    'verbose': 0,               # 0 for silent mode
}
# Train the model
num_round = 100  # Number of boosting rounds
bst = lgb.train(params, train_data, num_round, valid_sets = [test_data])
# Make predictions on the test set
y_pred_prob_lgb = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_lgb = [1 if pred > 0.5 else 0 for pred in y_pred_prob_lgb]  # Convert probabilities to binary predictions

#Modle evaluation 
accuracy = accuracy_score(y_test, y_pred_lgb)
roc_auc = roc_auc_score(y_test, y_pred_prob_lgb)

print(f'Accuracy on Test Set: {accuracy:.4f}')
print(f'ROC AUC on Test Set: {roc_auc:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred_lgb))

# %%
