

# run the below command to train a baseline model using train-small, validation and test datasets.
# the -m tag stands for mode which includes dnn (Deep Neural Network), xgboost (XGBoost), svm (Support Vector Machine)


python main.py -trd train-small -vd validation -tsd test -p bert-large -b -m dnn

#to run the gsg model on linkprediction task
python main.py -trd train-small -vd validation -tsd test -p bert-large -g -gnn gsg -tsk linkpred

#to finetune BERT + GNN
python main.py -trd train-small -vd validation -tsd test -ftg -tsk linkpred