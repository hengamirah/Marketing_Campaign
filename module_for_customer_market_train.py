# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:49:55 2022

@author: Amirah Heng
"""


import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import scipy.stats as ss
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization

#%% CLASSES & FUNCTIONS

class EDA():
    def __init__(self):
        pass
    
    def plot_data (self, con_column,cat_column,target,df):
        '''
        Parameters
        ----------
        Generate distribution plot for numerical columns in a dataset.
        Generate count plot for categorical features in a dataset.
        Parameters
        ----------
        con_column : list
            List of continuous features.
            
        cat_column : list
            List of categorical features.
            
        target : string
            Dataset contain 'term_deposit_subscribed'.
        df : DataFrame
            DataFrame of whole dataset.

        Returns
        -------
        Distribution and bar plots of columns names provided in argument..

        '''
        
        for con in con_column:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
        
        for cat in cat_column:
            plt.figure(figsize=(15,10))
            sns.countplot(df[cat])
            plt.show()
        
        for cat in cat_column:
            df.groupby([cat,target]).agg({target:'count'}).plot(kind='bar',figsize=(10,5))
            
class DataPreprocessing():
    def __init__(self):
        pass
    
    def label_encoder_data(self, cat_column,df,pickle_path):
        '''
        Generate label encoded data for categorical data
    
        Parameters
        ----------
        cat_column : list
            List of categorical features.
        df :  DataFrame
            DataFrame of whole dataset.
    
        Returns
        -------
        df : DataFrame
            DataFrame of whole dataset after label encoded.
    
        '''
        le = LabelEncoder()
        for index,i in enumerate(cat_column):
            temp=df[i]
            temp[temp.notnull()]=le.fit_transform(temp[temp.notnull()])
            df[i]=pd.to_numeric(temp,errors='coerce')
        
        #saving pickle file
        for index,i in enumerate(pickle_path):
            with open(pickle_path[index],'wb') as file:
                pickle.dump(le,file)
        
        return df
    
    
    def cramers_corrected_stat(self, confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
class ModelCreation():
    def __init__(self):
        pass
    def two_layer_model(self, nb_features,nb_class,nodenum=32,drop_rate=0.2):
        '''
        Generate a Neural Network with 2 hidden layers.

        Parameters
        ----------
        nb_features : tuple
            .list of selected features 
        nb_class : int
            Targeted data values from: 'term_deposit_subscribed'.
        nodenum : TYPE, number of nodes
        . The default is 32.
        drop_rate : TYPE, droputout rate 
        The default is 0.2.

        Returns
        -------
        model to be evaluated.

        '''
        model=Sequential() # to create container
        model.add(Input(shape=nb_features)) # to add input layer
        model.add(Dense(nodenum,activation='linear',name='HiddenLayer1')) # hidden layer 1
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(nodenum,activation='linear',name='HiddenLayer2')) # hidden layer 2
        model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        model.add(Dense(nb_class,activation='softmax',name='OutputLayer')) # output layer
        #use activation function: linear because data contains -ve value in balance
        model.summary()
        
        return model


class ModelEvaluation():       
    def __init__(self):
        pass
    
    def plot_evaluation(self,hist):
        '''
        Generate graphs to evaluate model loss and accuracy 

        Parameters
        ----------
        hist : TYPE
            model fitted with Training and Validation(test) dataset.

        Returns
        -------
        None.

        '''
        hist_keys = [i for i in hist.history.keys()]

        plt.figure()
        plt.plot(hist.history[hist_keys[0]]) #loss
        plt.plot(hist.history[hist_keys[2]]) #val loss
        plt.legend(['training_loss','validation_loss'])
        plt.show()

        plt.figure()
        plt.plot(hist.history[hist_keys[1]])
        plt.plot(hist.history[hist_keys[3]])
        plt.legend(['training_acc','validation_acc'])
        

    def model_evaluation(self, model,x_test,y_test):
        '''
        Generates confusion matrix and classification report based
        on predictions made by model using test dataset.

        Parameters
        ----------
        model : model
            Prediction model.
        x_test : ndarray
            Columns of test features.
        y_test : ndarray
            Target column of test dataset.
        label : list
            Confusion matrix labels.

        Returns
        -------
        Returns numeric report of model.evaluate(), 
        classification report and confusion matrix.

        '''
        result = model.evaluate(x_test,y_test)
        
        print(result) # loss, metrics
        y_pred=np.argmax(model.predict(x_test),axis=1)
        y_true=np.argmax(y_test,axis=1)
        print(y_true)
        print(y_pred)
        cm=confusion_matrix(y_true,y_pred)
        cr=classification_report(y_true, y_pred)
        print(cm)
        print(cr)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Reds)
        plt.show()
                





