
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import  RandomForestRegressor

def create_model (x_train, y_train, x_test) :
    pipeline_lasso_rf = Pipeline([
        ('poly', PolynomialFeatures(degree = 2, include_bias = False)),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(LassoCV(max_iter = 10000, cv = 5))),
        ('random_forest', RandomForestRegressor())
    ])

    pipeline_lasso_rf.fit(x_train, y_train)
    y_pred = pipeline_lasso_rf.predict(x_test)

    return y_pred