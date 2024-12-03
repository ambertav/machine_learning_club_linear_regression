import streamlit as st
import pandas as pd

from utils.dataset_cleaning import clean_dataset_one, clean_dataset_two
from utils.correlation_matrix import plot_correlation_matrix

df_first = pd.read_csv('data/insurance_dataset1.csv')
df_second = pd.read_csv('data/insurance_dataset2.csv')

st.title('Analyzing the Data')

st.header('First Dataset')
st.write('Initial dataset format:')
st.write(df_first.head())

df_first = clean_dataset_one(df_first)

st.write('Data after processing:')
st.write(df_first.head())

st.write('Correlation matrix:')
plot_correlation_matrix(df_first)



st.header('Second Dataset')
st.write('Initial dataset format:')
st.write(df_second.head())

df_second = clean_dataset_two(df_second)

st.write('Data after processing:')
st.write(df_second.head())

st.write('Correlation matrix:')
plot_correlation_matrix(df_second)
    

    