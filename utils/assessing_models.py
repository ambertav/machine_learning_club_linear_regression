import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def model_summary(y_test, y_pred):

  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)

  r2 = r2_score(y_test, y_pred)

  st.write('Mean Squared Error: ', mse)
  st.write('Root Mean Squared Error: ', rmse)
  st.write('R-squared: ', r2)

  assessing_model(y_test, y_pred)


def assessing_model (y_test, y_pred) :
  plt.scatter(y_pred, y_test, color='blue', label='True vs Predicted')
  plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Ideal Fit')
  plt.title('Predicted vs True Values')
  plt.xlabel('Predicted Values')
  plt.ylabel('True Values')
  plt.legend()
  st.pyplot(plt)