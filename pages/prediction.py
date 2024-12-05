import streamlit as st
import pandas as pd
from utils.dataset_cleaning import clean_dataset_two, split_train_and_test
from utils.helpers import calculate_bmi
from pipelines import poly_scaler_lasso_rf


st.title('Predict Your Insurance Price')

age = st.number_input('age', min_value = 0, max_value = 120, step = 1, value = 0)
sex = st.radio('sex', options = ['female', 'male'])
weight = st.number_input('weight (lbs)', min_value = 0, max_value = 300, step = 1, value = 0)
height = st.number_input('height (inches)', min_value = 0, max_value = 250, step=1, value = 0)
hereditary_diseases = st.radio('do you have any hereditary diseases', options=['yes','no'])
no_of_dependents = st.number_input('number of Dependents', min_value = 0, max_value = 20, step = 1, value=0)
smoker = st.radio('do you smoke?', options = ['yes', 'no'])
blood_pressure = st.number_input('blood pressure (mmHg)', min_value = 0, max_value = 300, step = 1, value = 120)
diabetes = st.radio('do you have diabetes?', options = ['yes','no'])
regular_ex = st.radio('do you exercise regularly?', options = ['yes', 'no'])

bmi = calculate_bmi(weight, height)

input_data = {
    'age': [age],
    'weight': [weight],
    'bmi': [bmi],
    'no_of_dependents': [no_of_dependents],
    'bloodpressure': [blood_pressure],
    'diabetes': [1 if diabetes == 'yes' else 0],
    'regular_ex': [1 if regular_ex == 'yes' else 0],
    'sex_male': [1 if sex == 'male' else 0],
    'disease_True': [1 if hereditary_diseases == 'yes' else 0],
    'smoker_1': [1 if smoker == 'yes' else 0],
}

if st.button('Submit') :
    second_df = pd.read_csv('data/insurance_dataset2.csv')
    second_df = clean_dataset_two(second_df)
    x2_train, x2_test, y2_train, y2_test = split_train_and_test(second_df, 'claim')

    model = poly_scaler_lasso_rf.create_model(x2_train, y2_train, 'dataset_2')

    input_df = pd.DataFrame(input_data)
    predicted_claim = model.predict(input_df)

    st.header('Prediction and Analysis')
    st.write('Input data overview:')
    st.write(input_df.head())
    st.write(f'Predicted Insurance Claim: ${predicted_claim[0]:.2f}')

    z_score = (predicted_claim[0] - y2_train.mean()) / y2_train.std()
    st.write(f'Predicted claim is {z_score:.2f} standard deviations away from the mean')
    
    if z_score > 2:
        st.write('This predicted claim is above the 95th percentile of the claims data (unusually high)')
    elif z_score < -2:
        st.write('This predicted claim is below the 5th percentile of the claims data (unusually low)')
    else:
        st.write('This predicted claim is within the typical range of the dataset')






