#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy as sa
import urllib
from PIL import Image
import scipy.stats as stats
from pandas.api.types import CategoricalDtype
from itertools import combinations
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

bestandnaam= "data" +".xlsx"
data = pd.read_excel (bestandnaam) #for an earlier version of Excel, you may need to use the file extension of 'xls'


# In[2]:


def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score

data_cleaned = data.drop("Laatste_LDL", axis=1)
y = data['Laatste_LDL']

x_train, x_test, y_train, y_test = train_test_split(data_cleaned, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
label_encoder.fit(['Atorva 40','Atorva 80','Atorva ander H','Rosuva 20', 'Rosuva 40','Rosuva ander H','Atorva 10','Atorva 20',"Atorva ander L","Rosuva 5","Rosuva 10","Rosuva ander L",'Simva 20','Simva 40','Simva 80','Simva ander','Prava 20','Prava 40','Prava ander','Ander','GeenINTOL','Geen'])
LabelEncoder()
list(label_encoder.classes_)

x_train['STATT0_resumee'] = label_encoder.transform(x_train['STATT0_resumee'])
x_test['STATT0_resumee'] = label_encoder.transform(x_test['STATT0_resumee'])
x_train['Laatste_statin'] = label_encoder.transform(x_train['Laatste_statin'])
x_test['Laatste_statin'] = label_encoder.transform(x_test['Laatste_statin'])

#save label encoder classes
np.save('classes.npy', label_encoder.classes_)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")
pred = best_xgboost_model.predict(x_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)
#%%
loaded_encoder = LabelEncoder()
loaded_encoder.classes_ = np.load('classes.npy',allow_pickle=True)
input_STATT0_resumee = loaded_encoder.transform(np.expand_dims("Geen",-1))
input_Laatste_statin = loaded_encoder.transform(np.expand_dims("Atorva 40",-1))

inputs = np.expand_dims([57,1,1,0,int(input_STATT0_resumee),int(input_Laatste_statin),1,0,0,0,1.8],0)
prediction = best_xgboost_model.predict(inputs)
print("final pred", np.squeeze(prediction,-1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




