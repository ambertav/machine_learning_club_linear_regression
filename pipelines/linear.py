import streamlit as st
from sklearn.linear_model import LinearRegression

@st.cache_data
def create_model (x_train, y_train) :
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    return lin_reg

@st.cache_data
def create_model_and_predict(x_train, y_train, x_test) :
    lin_reg = create_model(x_train, y_train)
    y_pred = lin_reg.predict(x_test)

    return y_pred
