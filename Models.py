from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


#Logistic Regression


def run_logistic_regression(train, val, test, feature_cols, target_col):

    required_cols = feature_cols + [target_col]

    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()
    test_clean  = test.dropna(subset=required_cols).copy()

    # Check if data disappeared
    if len(train_clean) == 0 or len(val_clean) == 0 or len(test_clean) == 0:
        raise ValueError("After dropping NaNs, one of the datasets became empty.")

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values

    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_col].values

    X_test = test_clean[feature_cols].values
    y_test = test_clean[target_col].values


    # standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)


    # train
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)


    # evalueate
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)

    test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)

    print("Validation Accuracy:", val_acc)
    print("Test Accuracy:", test_acc)

    return model, val_acc, test_acc

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