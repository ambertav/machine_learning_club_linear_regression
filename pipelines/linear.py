from sklearn.linear_model import LinearRegression

def create_model (x_train, y_train, x_test) :
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    y_pred = lin_reg.predict(x_test)

    return y_pred
