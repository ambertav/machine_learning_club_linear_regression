import streamlit as st
import pandas as pd

from utils.dataset_cleaning import clean_dataset_one, clean_dataset_two, split_train_and_test
from utils.assessing_models import model_summary
from pipelines import linear, poly_lasso, poly_scaler_lasso_rf

st.title('Models')

#Thinking we could split into 3 parts: 
#Feature selection/Scaling
#Models: Lin Reg, Poly Reg, Ridge/Lasso Reg, Elastic net, RF, RF Regressor
#Pipelines

first_df = pd.read_csv('data/insurance_dataset1.csv')
first_df = clean_dataset_one(first_df)
x1_train, x1_test, y1_train, y1_test = split_train_and_test(first_df, 'charges')

# Linear Regression
st.header('Linear Regression')
y1_pred = linear.create_model(x1_train, y1_train, x1_test)
model_summary(y1_test, y1_pred)


# Poly and LassoCV
st.header('Pipeline with Polynomial Features, LassoCV')
y1_pred = poly_lasso.create_model(x1_train, y1_train, x1_test)
model_summary(y1_test, y1_pred)


# Poly Scaler LassoCV RF
st.header('Pipeline with Polynomial Features, StandardScaler, LassoCV, Random Forest')
y1_pred = poly_scaler_lasso_rf.create_model(x1_train, y1_train, x1_test)
model_summary(y1_test, y1_pred)

st.header('Second Dataset')
st.header('Pipeline with Polynomial Features, StandardScaler, LassoCV, Random Forest')
second_df = pd.read_csv('data/insurance_dataset2.csv')
second_df = clean_dataset_two(second_df)
x2_train, x2_test, y2_train, y2_test = split_train_and_test(second_df, 'claim')
y2_pred = poly_scaler_lasso_rf.create_model(x2_train, y2_train, x2_test)
model_summary(y2_test, y2_pred)

