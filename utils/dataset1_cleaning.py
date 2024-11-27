import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data (df) :
    df.drop_duplicates(inplace = True)

    df_updated = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first = True)
    bool_columns = df_updated.select_dtypes(include='bool').columns
    df_updated[bool_columns] = df_updated[bool_columns].astype(int)

    return df_updated


def split_train_and_test (df) :
    input = df.drop(columns = ['charges'])
    y = df['charges']

    x_train, x_test, y_train, y_test = train_test_split(input, y, test_size = 0.2, random_state = 42)
    return x_train, x_test, y_train, y_test