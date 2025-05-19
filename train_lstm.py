import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
import argparse
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_lstm_model(seqs,labels,epochs=10,batch_size=32):
    X=seqs.reshape(-1,seqs.shape[1],1)
    y=labels
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler()
    X_tr=sc.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    X_te=sc.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    model=Sequential([LSTM(50,input_shape=(X_tr.shape[1],1)),Dense(len(np.unique(y)),activation='softmax')])
    model.compile('sparse_categorical_crossentropy','adam',['accuracy'])
    model.fit(X_tr,y_train,validation_data=(X_te,y_test),epochs=epochs,batch_size=batch_size)
    os.makedirs('models',exist_ok=True)
    path=os.path.join('models','lstm_model.h5')
    model.save(path)
    print("Saved LSTM to",path)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sequences',required=True)
    p.add_argument('--labels',required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=32)
    args=p.parse_args()
    seqs=np.load(args.sequences)
    labs=np.load(args.labels)
    train_lstm_model(seqs,labs,args.epochs,args.batch_size)