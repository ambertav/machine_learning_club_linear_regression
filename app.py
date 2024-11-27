import streamlit as st

import streamlit as st
from pages import view_data

st.markdown('''
# Machine Learning Club: Baruch College
## Linear Regression: Predicting Insurance Prices
            
#### Researchers:
Juan __ \\
Mahad __ \\
Jeremy Khusial \\
Himanshu Sharma \\
Suvedei Soyol-Erdene \\
Amber Taveras 
            
#### Product Manager:
James Witte-Cook
''')

st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Data', 'Models', 'Prediction'])

if page == 'Data':
    view_data.app()