import pandas as pd
from sklearn.model_selection import train_test_split

def clean_dataset_one (df) :
    df.drop_duplicates(inplace = True)

    df_updated = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first = True)
    bool_columns = df_updated.select_dtypes(include='bool').columns
    df_updated[bool_columns] = df_updated[bool_columns].astype(int)

    return df_updated


# Authored by: Suvedei Soyol-Erdene
def clean_dataset_two (data) :
    # data formatting / cleaning
    data.drop_duplicates(inplace = True)
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