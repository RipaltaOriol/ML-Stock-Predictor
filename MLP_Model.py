from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product
from sklearn.metrics import classification_report
import pandas as pd


def run_mlp(
    train, val, feature_cols, target_col,
    hidden_units=[64, 32],
    activation="relu",
    dropout_rate=0.0,
    learning_rate=1e-3,
    l2_reg=0.0,
    batch_size=256,
    epochs=50
):
    required_cols = feature_cols + [target_col]
    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()

    # transform
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clean[feature_cols])
    X_val   = scaler.transform(val_clean[feature_cols])

    y_train = train_clean[target_col].values.astype("float32")
    y_val   = val_clean[target_col].values.astype("float32")

    # binary
    model = keras.Sequential()
    model.add(layers.Input(shape=(len(feature_cols),)))

    for units in hidden_units:
        model.add(
            layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            )
        )
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Binary classification output
    model.add(layers.Dense(1, activation="sigmoid"))

    #model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc")
        ]
    )

    # train
    # implement early stopping
    es = keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor="val_loss"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[es],
        verbose=1
    )

    return model, scaler, history


def get_param_combinations(grid):
    keys = grid.keys()
    vals = grid.values()
    for combo in product(*vals):
        yield dict(zip(keys, combo))


def train_and_eval_mlp(train, val, FEATURES, TARGET, params):
    model, scaler, history = run_mlp(
        train=train,
        val=val,
        feature_cols=FEATURES,
        target_col=TARGET,
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
        l2_reg=params["l2_reg"],
        epochs=25   # limit training time
    )

    # Predict on validation
    val_clean = val.dropna(subset=FEATURES + [TARGET])
    X_val_scaled = scaler.transform(val_clean[FEATURES])
    y_val = val_clean[TARGET].values

    y_val_proba = model.predict(X_val_scaled).reshape(-1)
    auc_val = roc_auc_score(y_val, y_val_proba)

    return auc_val, model, scaler

def run_optimize_eval_MLP(train, val,test, FEATURES, TARGET):

    hyper_grid = {
    "hidden_units": [
        [32], 
        [64, 32],
        [128, 64, 32]
    ],
    "dropout_rate": [0.0, 0.1, 0.2],
    "learning_rate": [1e-3, 5e-4],
    "l2_reg": [0.0, 1e-4, 1e-3]
    }

    best_auc = -np.inf
    best_params = None
    best_model = None
    best_scaler = None

    for params in get_param_combinations(hyper_grid):
        #print(f"Training with params: {params}")

        auc_val, model, scaler = train_and_eval_mlp(
            train, val, FEATURES, TARGET, params
        )

        #print(f" -> Val AUC = {auc_val:.4f}")

        if auc_val > best_auc:
            best_auc = auc_val
            best_params = params
            best_model = model
            best_scaler = scaler

    #print("\nBest validation AUC:", best_auc)
    #print("Best params:", best_params)

    train_val = pd.concat([train, val], axis=0)

    model_final, scaler_final, _ = run_mlp(
        train=train_val,
        val=test,                # dummy val argument (ignored)
        feature_cols=FEATURES,
        target_col=TARGET,
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"],
        l2_reg=best_params["l2_reg"],
        pochs=25
    )

    test_clean = test.dropna(subset=FEATURES + [TARGET])

    X_test_scaled = scaler_final.transform(test_clean[FEATURES])
    y_test = test_clean[TARGET].values
    y_test_proba = model_final.predict(X_test_scaled).reshape(-1)
    y_test_pred  = (y_test_proba >= 0.5).astype(int)

    print("Test AUC:", roc_auc_score(y_test, y_test_proba))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))




