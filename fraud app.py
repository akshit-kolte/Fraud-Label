import pandas as pd
import numpy as np
import joblib
import streamlit as st
#Load the Model

model=joblib.load(open("decision_tree_model (1).joblib", 'rb'))

st.title ("Fraud Label app")
#Input feature
daysintopolicy = st.number_input("Days Into Policy",min_value=0.0)
daystoexpiry = st.number_input("Days To Expiry",min_value=0.0)
claimamountlakh = st.number_input("Claim Amount Lakh",min_value=0.0)
priorclaims = st.number_input("Prior Claims",min_value=0.0)
onlinechannel = st.number_input("Online Channel",min_value=0.0)
#Make Pred
if st.button('Fraud Label Prediction'):
	input_data=np.array([[daysintopolicy, daystoexpiry, claimamountlakh, priorclaims, onlinechannel]])
	prediction=model.predict(input_data)[0]

	st.success(f'Fraud Label:{prediction:.2f}')






