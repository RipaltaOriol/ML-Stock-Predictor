import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from itertools import product
import pandas as pd
import numpy as np

# --- Define the MLP model ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=[64, 32], activation="relu", dropout_rate=0.0, l2_reg=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError("Unsupported activation")
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = units

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.l2_reg = l2_reg

    def forward(self, x):
        return self.model(x)

# --- Train the MLP ---
def run_mlp_pytorch(train, val, feature_cols, target_col,
                    hidden_units=[64, 32], activation="relu",
                    dropout_rate=0.0, learning_rate=1e-3, l2_reg=0.0,
                    batch_size=256, epochs=50, patience=5, device="cpu"):

    # Prepare data
    required_cols = feature_cols + [target_col]
    train_clean = train.dropna(subset=required_cols).copy()
    val_clean   = val.dropna(subset=required_cols).copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_clean[feature_cols])
    X_val   = scaler.transform(val_clean[feature_cols])

    y_train = train_clean[target_col].values.astype("float32").reshape(-1, 1)
    y_val   = val_clean[target_col].values.astype("float32").reshape(-1, 1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MLP(input_dim=X_train.shape[1],
                hidden_units=hidden_units,
                activation=activation,
                dropout_rate=dropout_rate,
                l2_reg=l2_reg).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    best_val_loss = np.inf
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, scaler

# --- Predict function ---
def predict_proba_pytorch(model, scaler, df, feature_cols, device="cpu"):
    df_clean = df.dropna(subset=feature_cols).copy()
    X = scaler.transform(df_clean[feature_cols])
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        proba = model(X_t).cpu().numpy().reshape(-1)
    df_clean["proba"] = proba
    return df_clean

# --- Hyperparameter combination ---
def get_param_combinations(grid):
    keys = grid.keys()
    vals = grid.values()
    for combo in product(*vals):
        yield dict(zip(keys, combo))

# --- Train and evaluate MLP ---
def train_and_eval_mlp_pytorch(train, val, FEATURES, TARGET, params, device="cpu"):
    model, scaler = run_mlp_pytorch(
        train=train, val=val,
        feature_cols=FEATURES,
        target_col=TARGET,
        hidden_units=params["hidden_units"],
        dropout_rate=params["dropout_rate"],
        learning_rate=params["learning_rate"],
        l2_reg=params["l2_reg"],
        epochs=25,
        device=device
    )

    val_clean = val.dropna(subset=FEATURES + [TARGET])
    X_val_scaled = scaler.transform(val_clean[FEATURES])
    y_val = val_clean[TARGET].values

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        y_val_proba = model(X_val_t).cpu().numpy().reshape(-1)

    auc_val = roc_auc_score(y_val, y_val_proba)
    return auc_val, model, scaler

# --- Full optimization & evaluation ---
def run_optimize_eval_MLP_pytorch(train, val, test, FEATURES, TARGET, device="cpu"):
    hyper_grid = {
        "hidden_units": [[32], [64,32], [128,64,32]],
        "dropout_rate": [0.0, 0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "l2_reg": [0.0, 1e-4, 1e-3]
    }

    best_auc = -np.inf
    best_params = None
    best_model = None
    best_scaler = None

    for params in get_param_combinations(hyper_grid):
        auc_val, model, scaler = train_and_eval_mlp_pytorch(train, val, FEATURES, TARGET, params, device=device)
        if auc_val > best_auc:
            best_auc = auc_val
            best_params = params
            best_model = model
            best_scaler = scaler

    # Train on train+val
    train_val = pd.concat([train, val], axis=0)
    final_model, final_scaler = run_mlp_pytorch(
        train=train_val, val=test,   # dummy val
        feature_cols=FEATURES,
        target_col=TARGET,
        hidden_units=best_params["hidden_units"],
        dropout_rate=best_params["dropout_rate"],
        learning_rate=best_params["learning_rate"],
        l2_reg=best_params["l2_reg"],
        epochs=25,
        device=device
    )

    # Predict on test
    test_clean = test.dropna(subset=FEATURES + [TARGET])
    X_test_scaled = final_scaler.transform(test_clean[FEATURES])
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    final_model.eval()
    with torch.no_grad():
        y_test_proba = final_model(X_test_t).cpu().numpy().reshape(-1)

    y_test_pred = (y_test_proba >= 0.5).astype(int)

    print("Test AUC:", roc_auc_score(test_clean[TARGET].values, y_test_proba))
    print("Test accuracy:", accuracy_score(test_clean[TARGET].values, y_test_pred))
    print(classification_report(test_clean[TARGET].values, y_test_pred))

    return final_model, final_scaler, best_params
