#code is based on the given baseline code
import pandas as pd
import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras import optimizers, losses, activations, models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from src.data.load_data import importData

def get_CNN_model(dataset):
      #similar code to baseline
      nclass = 5
      inp = Input(shape=(187, 1))
      img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
      img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
      img_1 = MaxPool1D(pool_size=2)(img_1)
      img_1 = Dropout(rate=0.1)(img_1)
      img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
      img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
      img_1 = MaxPool1D(pool_size=2)(img_1)
      img_1 = Dropout(rate=0.1)(img_1)
      img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
      img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
      img_1 = GlobalMaxPool1D()(img_1)
      img_1 = Dropout(rate=0.2)(img_1)

      dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
      dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
      if dataset == "mitbih":
          dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)
      else:
          dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

      model = models.Model(inputs=inp, outputs=dense_1)
      opt = tf.keras.optimizers.Adam(0.001)

      #uncomment for correct dataset
      if dataset == "mitbih":
          model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
      else:
          model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
      model.summary()
      return model



def run_cnn(dataset):
  (X, Y, X_test, Y_test) = importData("/src/data", dataset)
  model = get_CNN_model()
  file_path = "baseline_cnn_mitbih.h5"
  checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
  redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
  csv_logger = CSVLogger("model_history_log.csv", append=True)
  callbacks_list = [checkpoint, early, redonplat, csv_logger]

  model.fit(X, Y, epochs=100, verbose=1, callbacks=callbacks_list, validation_split=0.1)
  model.load_weights(file_path)

  if dataset == "mitbih":
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)
    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : %s "% f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s "% acc)
  else:
    pred_test = model.predict(X_test)
    pred_test = (pred_test>0.5).astype(np.int8)
    f1 = f1_score(Y_test, pred_test)
    print("Test f1 score : %s "% f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s "% acc)
    
    print(roc_auc_score(Y_test,pred_test))
    precision, recall, _ = precision_recall_curve(Y_test, pred_test)
    print(auc(recall, precision))

    auroc = roc_curve(Y_test, pred_test)

    df = pd.DataFrame(list(zip(auroc[0],auroc[1],precision,recall)))
    df.to_csv("_AUROC_AUPRC1" + dataset + '.csv') 

run_cnn(dataset="mitbih")
