# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:19:26 2022

There are a few essential aspects of the marketing campaign namely, 
customer segmentation, promotional strategy, and etc. Correctly identified strategy may help to expand and grow the bank’s revenue.

1) Develop a deep learning model using TensorFlow which only comprises of Dense, Dropout, and Batch Normalization layers.
2) The accuracy of the model must be more than 70%.
3) Display the training loss and accuracy on TensorBoard
4) Create modules (classes) for repeated functions to ease your training and testing process

@author: Amirah Heng
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from module_for_customer_market_train import EDA, DataPreprocessing
from module_for_customer_market_train import ModelCreation, ModelEvaluation
#EDA
#%% STEP 1) Statics
CSV_PATH =os.path.join(os.getcwd(),'Market_Train.csv')
JOB_TYPE_PICKLE_PATH=os.path.join(os.getcwd(),'model','job.pkl')
MARITAL_PICKLE_PATH=os.path.join(os.getcwd(),'model','marital.pkl')
EDUCATION_PICKLE_PATH=os.path.join(os.getcwd(),'model','education.pkl')
DEFAULT_PICKLE_PATH=os.path.join(os.getcwd(),'model','default.pkl')
HOUSING_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','housing_loan.pkl')
PERSONAL_LOAN_PICKLE_PATH=os.path.join(os.getcwd(),'model','personal_loan.pkl')
COMMUNICATION_TYPE_PICKLE_PATH=os.path.join(os.getcwd(),'model','communication_type.pkl')
MONTH_PICKLE_PATH=os.path.join(os.getcwd(),'model','month.pkl')
PREV_CAMPAIGN_OUTCOME_PICKLE_PATH=os.path.join(os.getcwd(),'model','prev_campaign_outcome.pkl')
OHE_PATH =os.path.join(os.getcwd(),'model','ohe.pkl')
SS_PATH =os.path.join(os.getcwd(),'model','ss.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'log',log_dir)
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','model.h5')
#%% STEP 1) DATA LOADING
df = pd.read_csv(CSV_PATH)

#%% STEP 2) DATA INSPECTION

df.info() #to see NaNs and datatype
df=df.drop(labels='id',axis=1)
column_names= ['customer_age', 'job_type', 'marital', 'education', 'default','balance', 
               'housing_loan', 'personal_loan', 'communication_type', 'day_of_month', 
               'month', 'last_contact_duration', 'num_contacts_in_campaign', 
               'num_contacts_prev_campaign', 'prev_campaign_outcome', 
               'term_deposit_subscribed']
               
#describe column from categorical and continuous data
target = 'term_deposit_subscribed'  #target selected as term_deposit_subscribed

#continuous data column
con_column=['customer_age', 'balance', 'day_of_month', 'last_contact_duration',
            'num_contacts_in_campaign', 'days_since_prev_campaign_contact',
            'num_contacts_prev_campaign'] 

#categorical data column
cat_column=['job_type','marital','education','default','housing_loan',
            'personal_loan','communication_type','month','prev_campaign_outcome'] 

#plot data
EDA().plot_data(con_column,cat_column,target,df)

#check for NaN values
df.isna().any()
df.isna().sum()# customer_age:619 marital:150, balance: 399, personal_loan:149
# last_contact_duration:311, num_contacts_in_campaign :112, days_since_prev_campaign_contact :25831

df.count() #31,647 max data
#since the NaN values is about 80% of the data, days_since_prev_campaign_contact will be removed

temp= df.describe().T #balance has many outliers

#%% STEP 3) DATA CLEANING

#remove days_since_prev_campaign_contact column,since the NaN values is about 80% of the data
df= df.drop(labels='days_since_prev_campaign_contact',axis=1)
con_column.remove('days_since_prev_campaign_contact')

#label encoder categorical data
#save label encoder pickle 
pickle_path=[JOB_TYPE_PICKLE_PATH,
             MARITAL_PICKLE_PATH,
             EDUCATION_PICKLE_PATH,
             DEFAULT_PICKLE_PATH,
             HOUSING_LOAN_PICKLE_PATH,
             PERSONAL_LOAN_PICKLE_PATH,
             COMMUNICATION_TYPE_PICKLE_PATH,
             MONTH_PICKLE_PATH,
             PREV_CAMPAIGN_OUTCOME_PICKLE_PATH]

dp=DataPreprocessing()
dp.label_encoder_data(cat_column,df,pickle_path)
df.isna().sum() #NaN values remain same

#use iterative imputer to fill NaN values
ii = IterativeImputer()
df_II = ii.fit_transform(df)
df= pd.DataFrame(df_II)
df.columns= column_names

 #round of data to integer value
for index, i in enumerate(column_names):
    df[i]= np.floor(df[i]).astype('int')

#check for NaN values
df.isna().any()
df.isna().sum() #no More Nan values

#%% STEP 4) FEATURES SELECTION

logreg=LogisticRegression()

#Logistic Regression to determine accuracy score between continuous data and categorical data
for con in con_column:
    logreg.fit(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed'])
    print(con,':',logreg.score(np.expand_dims(df[con],axis=-1),df['term_deposit_subscribed']))

#Cramer's V to determine accuracy score between categorical data and categorical data
for cat in cat_column:
    confusion_matriX = pd.crosstab(df[cat], df['term_deposit_subscribed']).to_numpy()
    print('{}: accuracy is {}'.format(cat,dp.cramers_corrected_stat(confusion_matriX)))
    
#All continuous data has high correlation value(>0.8) with target term_deposit_subscribed
#Hence is selected as Features -> X
X = df[con_column]
y = df['term_deposit_subscribed']

#%% STEP 5) DATA PRE-PROCESSING

Std_scaler= StandardScaler()
scaled_X= Std_scaler.fit_transform(X)

#Use OneHotEncoder for target since term_deposit_subscribed is categorical
ohe= OneHotEncoder(sparse=False)
y=ohe.fit_transform(np.expand_dims(y,axis=-1))

#train test split data
x_train,x_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=123)

#%% STEP 6) MODEL DEVELOPMENT
md=ModelCreation()
nb_features= np.shape(X)[1:]
nb_class= len(np.unique(y))
model=md.two_layer_model(nb_features,nb_class,nodenum=32)

# Visualising Neural Network model
plot_model(model,show_shapes=True, show_layer_names=(True))

# Compile model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

#Callbacks
tensorboard_callback=TensorBoard(log_dir=LOG_FOLDER_PATH)
early_stopping_callback = EarlyStopping(monitor='loss',patience=5)

#%% STEP 7) MODEL ANALYSIS

# Model Fitting and Testing
hist = model.fit(x=x_train,y=y_train,batch_size=64,epochs=100,
                validation_data=(x_test,y_test),
                callbacks=[tensorboard_callback,early_stopping_callback])
#accuracy is 90%

#%% STEP 8) MODEL EVALUATION
me=ModelEvaluation()
#Plot hist evaluation
me.plot_evaluation(hist)    

#Model evaluation
me.model_evaluation(model, x_test, y_test)
#%% STEP 9) MODEL SAVING
#saving ohe model
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
#save std scaler model
with open(SS_PATH,'wb') as file:
    pickle.dump(Std_scaler,file)
#to save Neural network model
model.save(MODEL_SAVE_PATH)

#%% Discussion

# The deep learning model achieved a good performance with 90% accuracy
# showing 'customer_age', 'balance', 'day_of_month', 'last_contact_duration',
# 'num_contacts_in_campaign', 'days_since_prev_campaign_contact',
# 'num_contacts_prev_campaign'has the highest correlation (above 80%) with
# term_deposit_subscribed
# The test set can be improved by training more dataset into the model
# and reducing the underfitting model


