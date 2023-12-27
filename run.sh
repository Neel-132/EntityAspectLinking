

# run the below command to train a baseline model using train-small, validation and test datasets.
# the -m tag stands for mode which includes dnn (Deep Neural Network), xgboost (XGBoost), svm (Support Vector Machine)


python main.py -trd train-small -vd validation -tsd test -p bert-large -b -m dnn