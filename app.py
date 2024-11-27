import streamlit as st

page = ['Home', 'Data', 'Models']

if page == 'Home':
    st.query_params(page='home')
elif page == 'Data':
    st.query_params(page='data')
elif page == 'Models':
    st.query_params(page='models')


st.markdown('''
# Machine Learning Club: Baruch College
## Linear Regression: Predicting Insurance Prices
            
#### Researchers:
Juan \\
Mahad \\
Jeremy Khusial \\
Himanshu Sharma \\
Suvedei Soyol-Erdene \\
Amber Taveras 
            
#### Product Manager:
James Witte-Cook
''')