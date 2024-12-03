import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

@st.cache_data
def clean_dataset_one (df) :
    df_updated = df.drop_duplicates()

    df_updated = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first = True)
    bool_columns = df_updated.select_dtypes(include='bool').columns
    df_updated[bool_columns] = df_updated[bool_columns].astype(int)

    return df_updated


@st.cache_data
# Authored by: Suvedei Soyol-Erdene
def clean_dataset_two (data) :
    # data formatting / cleaning
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.copy()
    data["disease"] = data["hereditary_diseases"] != "NoDisease"
    df_updated = pd.get_dummies(data, columns=['sex', 'disease', 'smoker'], drop_first = True)
    bool_columns = df_updated.select_dtypes(include='bool').columns
    df_updated[bool_columns] = df_updated[bool_columns].astype(int)
    df_updated = df_updated.drop(columns = ["city", "hereditary_diseases" , "job_title"])

    return df_updated


def split_train_and_test (df, column_name) :
    input = df.drop(columns = [column_name])
    y = df[column_name]

    x_train, x_test, y_train, y_test = train_test_split(input, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test