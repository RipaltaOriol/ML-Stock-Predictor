import pandas as pd
import helper_functions as hf
import numpy as np


#####The point of this file is to create a model that guesses at random to use as baseline for our other models
####We will generate a uniform random variable z and if z > 0.5 we predict 1 and if z < 0.5 we predict 0

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

class RandomPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)

    def fit(self, X, y):
        # No model fitting needed
        return self

    def predict_proba(self, X):
        # Uniform random probabilities
        probs = np.random.uniform(0, 1, size=len(X))
        # Format like sklearn: return Nx2 array
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
def run_random_prediction_model(train, val, feature_cols, target_col, random_state=42):
    required_cols = feature_cols + [target_col]

    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values.astype(int)

    model = RandomPredictor(random_state=random_state)
    model.fit(X_train, y_train)

    return model

def eval_random(model, val, FEATURES, TARGET):
    val_clean = val.dropna(subset=FEATURES + [TARGET]).copy()

    X_val = val_clean[FEATURES].values
    y_val = val_clean[TARGET].values.astype(int)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_proba)

    return auc_val, y_val_proba


def run_random_baseline(train, val, test, FEATURES, TARGET, random_state=42):

    # Fit model using train
    model = run_random_prediction_model(
        train=train,
        val=val,
        feature_cols=FEATURES,
        target_col=TARGET,
        random_state=random_state
    )

    # validation
    val_clean = val.dropna(subset=FEATURES + [TARGET]).copy()
    X_val = val_clean[FEATURES].values
    y_val = val_clean[TARGET].values.astype(int)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred  = (y_val_proba >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, y_val_proba)
    val_acc = accuracy_score(y_val, y_val_pred)

    # test
    test_clean = test.dropna(subset=FEATURES + [TARGET]).copy()
    X_test = test_clean[FEATURES].values
    y_test = test_clean[TARGET].values.astype(int)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred  = (y_test_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_test_proba)
    test_acc = accuracy_score(y_test, y_test_pred)

    # return same metrics as the other models
    return val_auc, val_acc, test_auc, test_acc
