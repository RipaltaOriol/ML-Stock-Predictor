from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def run_random_forest(
    train, val, feature_cols, target_col,
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
):
    # Clean data
    required_cols = feature_cols + [target_col]
    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()

    # Features
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values.astype(int)

    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_col].values.astype(int)

    # Model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        ## added out of bag scoring
        oob_score=True
    )

    # Train
    rf.fit(X_train, y_train)

    return rf

def eval_rf(model, val, FEATURES, TARGET):
    val_clean = val.dropna(subset=FEATURES + [TARGET]).copy()

    X_val = val_clean[FEATURES].values
    y_val = val_clean[TARGET].values.astype(int)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = (y_val_proba >= 0.5).astype(int)
    auc_val = roc_auc_score(y_val, y_val_proba)
    acc_val = accuracy_score(y_val,y_pred)

    return auc_val, acc_val, y_val_proba

def rf_param_combinations(grid):
    keys = list(grid.keys())
    for combo in product(*grid.values()):
        yield dict(zip(keys, combo))


def run_optimize_eval_RF(train, val, test, FEATURES, TARGET):
    rf_grid = {
    "n_estimators": [400],
    "max_depth": [None, 10, 20],   # shallow vs deep comparison
    "max_features": ["sqrt", 0.3], 
    "min_samples_leaf": [1, 3],    # prevents overfitting
    "class_weight": ["balanced"]   # explicit class imbalance handling
    }

    best_auc = -np.inf
    best_acc = None
    best_params = None
    best_model = None

    for params in rf_param_combinations(rf_grid):
        #print(f"Training RF with params: {params}")

        model = run_random_forest(
            train=train,
            val=val,
            feature_cols=FEATURES,
            target_col=TARGET,
            **params
        )

        auc_val, acc_val, _ = eval_rf(model, val, FEATURES, TARGET)

        #print(f" -> Val AUC = {auc_val:.4f}")

        if auc_val > best_auc:
            best_auc = auc_val
            best_acc = acc_val
            best_params = params
            best_model = model

    #print("\nBest validation AUC:", best_auc)
    #print("Best params:", best_params)

    train_val = pd.concat([train, val], axis=0)

    final_rf = run_random_forest(
        train=train_val,
        val=test,       # dummy, not used
        feature_cols=FEATURES,
        target_col=TARGET,
        **best_params
    )

    test_clean = test.dropna(subset=FEATURES + [TARGET]).copy()
    X_test = test_clean[FEATURES].values
    y_test = test_clean[TARGET].values.astype(int)

    y_test_proba = final_rf.predict_proba(X_test)[:, 1]
    y_test_pred  = (y_test_proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_test_proba)
    test_acc = accuracy_score(y_test, y_test_pred)


    #print("Test AUC:", roc_auc_score(y_test, y_test_proba))
    #print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    print("OOB Score:", final_rf.oob_score_)

    #return best_auc, roc_auc_score(y_test, y_test_proba)

    importances = pd.Series(final_rf.feature_importances_, index=FEATURES)
    print("\nTop 10 Feature Importances:")
    print(importances.sort_values(ascending=False).head(10))

    return best_auc, best_acc, test_auc, test_acc





