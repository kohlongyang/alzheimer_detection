import numpy as np
import pandas as pd
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import matthews_corrcoef
import optuna


X_TRAIN_DIR = "processed_data/glcmX_train.pkl"
y_TRAIN_DIR = "processed_data/glcm/y_train.pkl"
X_TEST_DIR = "processed_data/glcm/X_test.pkl"
y_TEST_DIR = "processed_data/glcm/y_test.pkl"

X_train = pd.read_pickle(X_TRAIN_DIR)
y_train = pd.read_pickle(y_TRAIN_DIR)
X_test = pd.read_pickle(X_TEST_DIR)
y_test = pd.read_pickle(y_TEST_DIR)


def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = cb.CatBoostRegressor(**params, silent=True, allow_writing_files=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    rmse = mean_squared_error(y_train, predictions, squared=False)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

#edit params based on line 40
#params = {'learning_rate': 0.1, 'depth': 7, 'l2_leaf_reg': 5, 'iterations': 100, 'eval_metric': 'MultiClass'}
model = CatBoostClassifier(
    iterations = 100, 
    loss_function='MultiClass',
    bootstrap_type = "Bayesian",
    eval_metric = 'MultiClass',
    leaf_estimation_iterations = 100,
    random_strength = 0.5,
    depth = 7,
    l2_leaf_reg = 5,
    learning_rate=0.1,
    bagging_temperature = 0.5,
    allow_writing_files=False,
    ) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
accuracy = (y_pred == np.array(y_test)).mean() 
print("Test Accuracy:", accuracy) 
print(classification_report(y_test,y_pred))
print(matthews_corrcoef(y_test,y_pred))
