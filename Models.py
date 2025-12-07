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

    # in logistic regression we don't need validation since there are no hyperparameters so we can join X train and X val
    # X_train_full = np.concatenate([X_train, X_val], axis=0)
    # y_train_full = np.concatenate([y_train, y_val], axis=0)

    cvals        = [0.01, 0.1, 1.0, 10.0]
    class_weights   = [None, "balanced"]

    best_auc = -np.inf
    best_params = None

    for cval, cw in product(cvals, class_weights):
        clf = LogisticRegression(
            max_iter=500,
            C = cval,
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

        if curr_auc > best_auc:
            best_auc = curr_auc
            best_params = {"C": cval, "class_weight": cw}
        
    # merge train and validation sets to fit the final model
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    lr = LogisticRegression(
        max_iter=1000,
        C=best_params["C"],
        class_weight=best_params["class_weight"],
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", lr),
    ])

    model.fit(X_train_val, y_train_val)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= 0.5).astype(int)

    print("Test AUC:", roc_auc_score(y_test, y_prob))
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

# LSTM
def make_lstm_sequences(df, feature_cols, target_col, seq_len):
    X_list, y_list = [], []

    # Drop rows with NaNs in features or target
    df = df.dropna(subset=feature_cols + [target_col]).copy()

    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("date")

        feature_mat = tdf[feature_cols].values
        labels = tdf[target_col].values

        # Slide sequence window
        for i in range(seq_len, len(tdf)):
            X_list.append(feature_mat[i-seq_len:i])
            y_list.append(labels[i])

    return np.array(X_list), np.array(y_list)


def run_lstm(train, val, test, feature_cols, target_col, seq_len=30):
    
    # scale
    scaler = StandardScaler()

    train_scaled = train.copy()
    val_scaled   = val.copy()
    test_scaled  = test.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
    val_scaled[feature_cols]   = scaler.transform(val[feature_cols])
    test_scaled[feature_cols]  = scaler.transform(test[feature_cols])

    # sequence
    X_train, y_train = make_lstm_sequences(train_scaled, feature_cols, target_col, seq_len)
    X_val, y_val     = make_lstm_sequences(val_scaled, feature_cols, target_col, seq_len)
    X_test, y_test   = make_lstm_sequences(test_scaled, feature_cols, target_col, seq_len)

    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    #model
    model = models.Sequential([
        layers.Input(shape=(seq_len, len(feature_cols))),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=128,
        shuffle=False   # IMPORTANT for time series
    )

    #evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

    return model, history, test_acc