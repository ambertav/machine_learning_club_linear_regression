import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def plot_correlation_matrix (df) :
    # constructing and showing plot
    correlation_matrix = df.corr().round(2)
    plt.figure(figsize = (10, 8))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f") #0.2f, decimal
    plt.title('Correlation Matrix')
    st.pyplot(plt)