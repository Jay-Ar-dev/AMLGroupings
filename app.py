import streamlit as st
import pandas as pd
import numpy as np

st.title("🏠 Foreclosed Property – Minimum Bid Price Predictor")

lot_area = st.number_input("Lot Area (sqm)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)

#floor are na
floor_area = st.number_input("Floor Area (sqm)", min_value=0.0, max_value=10000.0, value=50.0, step=10.0)

#dropdown Region
region = st.selectbox("Region", options=["METRO MANILA", "NON-METRO MANILA"])

#remarks
remarks = st.selectbox("Remarks", options=["Unoccupied", "Occupied"]) 

#status of tct
status_tct = st.selectbox("Status of TCT", options=["TCT under the Bank", "For Title Consolidation"]) 

#then button predict, design button to color blue and make it full width
if st.button("Predict Minimum Bid Price", use_container_width=True, type="primary"):

    st.write("Predicting minimum bid price...")

    #bale computation nalang dito ang kulang
    
    # Here you would add the code to load your model, preprocess the input, and make a prediction
    # For demonstration, we'll just show a placeholder result

    #dito na part add mo nalang po ang computation sa prediction

    #Pag success ang computations, sa placeholder nalang po ilagay ang result
    st.success("Prediction completed successfully! (This is a placeholder result)")

