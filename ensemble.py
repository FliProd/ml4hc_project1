# code inspired by https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from numpy import dstack
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve
from src.data.load_data import importData

import os
 
# load models from file
def load_all_models(model_list):
    all_models = list()
    for i in model_list:
      try:
        model = load_model('.' + i)
        all_models.append(model)
      except:
        try:
          model = torch.load('.' + i)
          all_models.append(model)
        except:
          print('model not found')
    return all_models
 
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
      # make prediction
      #yhat = model.predict(inputX, verbose=0)
      try:
        X_test = torch.tensor(inputX, requires_grad=False).type(torch.float)
        #yhat = model(inputX[0])
        yhat = model(torch.permute(X_test, (0, 2, 1)))
        yhat = yhat.detach().numpy()
        yhat = yhat.argmax(axis=1)
      except:
        yhat = model.predict(inputX, verbose=0).flatten()
      # stack predictions into [rows, members, probabilities]
      if stackX is None:
        stackX = yhat
      else:
        stackX = dstack((stackX, yhat))
    
    #stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX[0]
 
#do logistic regression on the predictions from the models
def fit_stacked_model(members, inputX, inputy):
	stackedX = stacked_dataset(members, inputX)
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model
 
# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	stackedX = stacked_dataset(members, inputX)
	yhat = model.predict(stackedX)
	return yhat

#take average of the prediction from the models
def ensemble_average(model_list, X):
  yhat_ = None
  for model in model_list:
      try:
        X_test = torch.tensor(X, requires_grad=False).type(torch.float)
        yhat = model(torch.permute(X_test, (0, 2, 1)))
        yhat = yhat.detach().numpy()
        yhat = yhat.argmax(axis=1)
      except:
        yhat = model.predict(X, verbose=0).flatten()
      if yhat_ is None:
        try:
          yhat_ = model(torch.permute(X_test, (0, 2, 1)))
          yhat_ = yhat_.detach().numpy()
          yhat_ = yhat_.argmax(axis=1)
        except:
          yhat_ = model.predict(X, verbose=0).flatten()
      yhat_ =  np.mean( np.array([ yhat, yhat_ ]), axis=0 )


  return yhat_


model_list_ptbdb = ["/models/Autoencoder_ptbdb_1500_20_2", "/models/RNN_ptbdb_100_100_2", "/models/LSTM_ptbdb_1500_100_2", "/models/baseline_cnn_ptbdb.h5"]
model_list_mitbih = ["/models/LSTM_mitbih_100_100_2", "/models/Autoencoder_mitbih_100_30_2", "/models/RNN_mitbih_1500_100_2", "/models/cnn_mitbih.h5"]

#change following 2 lines to switch dataset
(_, _, X_test, Y_test) = importData("./data/raw/", "mitbih")
members = load_all_models(model_list_mitbih)

print("Loaded %d models" % len(members))

# fit stacked model using the ensemble
model = fit_stacked_model(members, X_test, Y_test)
# evaluate model on test set
yhat = stacked_prediction(members, model, X_test)
yhat_average = ensemble_average(members, X_test)


#comment/uncomment next two lines for right dataset
yhat_average = (yhat_average>0.5).astype(np.int8)
#yhat_average = np.argmax(yhat_average, axis=-1)

acc = accuracy_score(Y_test, yhat)
acc_average = accuracy_score(Y_test, yhat_average)

print('Logistic Regression Accuracy: %.3f' % acc)
print('Averaged Test Accuracy: %.3f' % acc_average)
try:
  print("Auroc Logistic regression: " + str(roc_auc_score(Y_test,yhat)))
  print("Auroc average: " + str(roc_auc_score(Y_test,yhat_average)))
  precision, recall, _ = precision_recall_curve(Y_test, yhat)
  print("Auprc logistic regression: " + str(auc(recall, precision)))
  precision, recall, _ = precision_recall_curve(Y_test, yhat_average)
  print("Auprc average: " + str(auc(recall, precision)))
except Exception as e:
  print(e)