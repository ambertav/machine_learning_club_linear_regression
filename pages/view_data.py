import streamlit as st
import pandas as pd

from utils.dataset1_cleaning import clean_data


def app () :
    df = pd.read_csv('data/insurance_dataset1.csv')

    st.header('Analyzing the Data')
    st.write('Data prior to cleaning:')
    st.write(df.head())

    df_updated = clean_data(df)


    st.write('Data after to cleaning:')
    st.write(df_updated.head())



    