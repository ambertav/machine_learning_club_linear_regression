import streamlit as st
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import  RandomForestRegressor

@st.cache_data
# Authored by: Amber Taveras
def create_model (x_train, y_train, dataset_name) :
    pipeline_lasso_rf = Pipeline([
        ('poly', PolynomialFeatures(degree = 2, include_bias = False)),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(LassoCV(max_iter = 10000, cv = 5))),
        ('random_forest', RandomForestRegressor())
    ])

    pipeline_lasso_rf.fit(x_train, y_train)

    return pipeline_lasso_rf

@st.cache_data
def create_model_and_predict(x_train, y_train, x_test, dataset_name) :
    pipeline_lasso_rf = create_model(x_train, y_train, dataset_name)
    y_pred = pipeline_lasso_rf.predict(x_test)
    return y_pred
