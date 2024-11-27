import streamlit as st
import pandas as pd

from utils.dataset1_cleaning import clean_data
from utils.correlation_matrix import plot_correlation_matrix


def app () :
    df = pd.read_csv('data/insurance_dataset1.csv')

    st.header('Analyzing the Data')
    st.write('Initial dataset format:')
    st.write(df.head())

    df = clean_data(df)

    st.write('Data after processing:')
    st.write(df.head())

    st.write('Correlation matrix:')
    plot_correlation_matrix(df)
    

    