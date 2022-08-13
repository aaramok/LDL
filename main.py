#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import date
import sqlalchemy as sa
import urllib
from PIL import Image
import scipy.stats as stats
from pandas.api.types import CategoricalDtype
from itertools import combinations
from hyperopt import hp
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

bestandnaam= "data" +".xlsx"
data = pd.read_excel (bestandnaam) #


# In[5]:


st.header("LDL prediction app")
st.text_input("Enter your Name: ", key="name")


# In[6]:


encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show dataframe'):
    data
    
st.subheader("Vul onderstaande in voor een LDL predictie")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio(
        'Huidig statine gebruik:',
        np.unique(data['STATT0_resumee']))
input_Length1 = st.slider('LDL bij baseline', 0.0, max(data["Admission_LDL"]), 7.0)


# In[ ]:





# In[ ]:




