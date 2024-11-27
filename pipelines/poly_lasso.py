import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def create_model (x_train, y_train, x_test) :
    pipeline_lasso = Pipeline([
        ('poly', PolynomialFeatures(degree = 2, include_bias = False)),
        ('lasso', LassoCV(alphas = np.logspace(-6, 6, 13), max_iter = 10000, cv = 5))
    ])

    pipeline_lasso.fit(x_train, y_train)
    y_pred = pipeline_lasso.predict(x_test)

    return y_pred