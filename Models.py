from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

import numpy as np
from itertools import product

import tensorflow as tf
from tensorflow.keras import layers, models

#Logistic Regression


def run_logistic_regression(train, val, test, feature_cols, target_col):

    # select features we are interested in
    required_cols = feature_cols + [target_col]

    # drop NaN values in selected features
    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()
    test_clean  = test.dropna(subset=required_cols).copy()
    

    # ensure we still have data
    if len(train_clean) == 0 or len(val_clean) == 0 or len(test_clean) == 0:
        raise ValueError("After dropping NaNs, one of the datasets became empty.")

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values

    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_col].values

    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values


    cvals = [0.01, 0.1, 1.0, 10.0]
    class_weights = [None, "balanced"]
    penalties = ["l2", None]

    best_auc = -np.inf
    best_acc = None
    best_params = None

    for cval, cw, pen in product(cvals, class_weights, penalties):
        clf = LogisticRegression(
            max_iter=500,
            C = cval,
            penalty = pen,
            class_weight = cw,
            solver = "lbfgs"
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_val)[:, 1]
        curr_auc = roc_auc_score(y_val, y_prob)

        # Convert probabilities to class predictions
        y_pred = (y_prob >= 0.5).astype(int)
        curr_acc = accuracy_score(y_val, y_pred)

        if curr_auc > best_auc:
            best_auc = curr_auc
            best_acc = curr_acc
            best_params = {"C": cval, "class_weight": cw, "penalty": pen}
        
    # merge train and validation sets to fit the final model
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    lr = LogisticRegression(
        max_iter=1000,
        C=best_params["C"],
        penalty=best_params["penalty"],
        class_weight=best_params["class_weight"],
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lr),
    ])

    model.fit(X_train_val, y_train_val)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= 0.5).astype(int)

    # print("Test AUC:", roc_auc_score(y_test, y_prob))
    # print("Test accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    
    coeffs = model.named_steps["clf"].coef_[0]                 
    intercept = model.named_steps["clf"].intercept_[0]
    # print("Coefficients: ", coeffs)
    
    # test_score = model.score(X_test, y_test)
    val_auc = best_auc
    test_auc = roc_auc_score(y_test, y_prob)
    test_acc = accuracy_score(y_test, y_pred)
    
    #return val_auc, test_auc
    return best_auc, best_acc, test_auc, test_acc, best_params
    
    
