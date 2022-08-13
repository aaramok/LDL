import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

bestandnaam= "data" +".xlsx"
data = pd.read_excel (bestandnaam) #

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




