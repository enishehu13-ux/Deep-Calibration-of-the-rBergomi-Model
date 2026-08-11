import numpy as np
import pandas as pd
import keras
from keras import layers
import keras.ops as ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Custom RMSE Loss Function 

@keras.saving.register_keras_serializable()
def rmse_loss(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))


# PyTorch-Style Print Callback

class PyTorchStyleLogger(keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.best_val_loss = float('inf')
        self.best_weights = None

    def _get_current_lr(self):
        lr_attr = self.model.optimizer.learning_rate
        if callable(lr_attr):
            lr_tensor = lr_attr(self.model.optimizer.iterations)
        else:
            lr_tensor = lr_attr
        return float(ops.convert_to_numpy(lr_tensor))

    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Track best validation loss (EarlyStopping will also restore best weights)
        current_val_loss = logs.get('val_loss')
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_weights = self.model.get_weights()

        t_mse = logs.get('mse', 0.0)
        t_rmse = logs.get('rmse', 0.0)
        t_mae = logs.get('mae', 0.0)
        t_mape = logs.get('mape', 0.0) 

        v_mse = logs.get('val_mse', 0.0)
        v_rmse = logs.get('val_rmse', 0.0)
        v_mae = logs.get('val_mae', 0.0)
        v_mape = logs.get('val_mape', 0.0) 

        current_lr = self._get_current_lr()

        print(f"Epoch [{epoch+1:03d}/{self.epochs}] | LR: {current_lr:.2e}")
        print(f"  Train -> MSE: {t_mse:.4e} | RMSE: {t_rmse:.4e} | MAE: {t_mae:.4e} | MAPE: {t_mape:.4f}%")
        print(f"  Test  -> MSE: {v_mse:.4e} | RMSE: {v_rmse:.4e} | MAE: {v_mae:.4e} | MAPE: {v_mape:.4f}%")
        print("-" * 75)


#  Load Data Directly from CSV

print("Loading data from CSV...")
csv_filename = "rbergomi_dataset.csv"

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_filename)

# Extra safety net: drop any completely empty or NaN rows
df = df.dropna()

print(f"Loaded {len(df)} total rows.")

# X gets all columns EXCEPT the 'IV' column
X = df.drop(columns=['IV']).values.astype(np.float32)

# y gets ONLY the 'IV' column, reshaped for Keras
y = df['IV'].values.astype(np.float32).reshape(-1, 1)


#  Split and Scale Data

print("Splitting and scaling data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, "rbergomi_scaler.pkl")


# 3. Build the Model

print("Building the model...")
input_dim = X_train.shape[1]

inputs = keras.Input(shape=(input_dim,), name="features")

# 4 Hidden layers with exactly 64 nodes each, ELU activation
x = layers.Dense(64, activation="swish", kernel_initializer='he_normal', name="dense_1")(inputs)
x = layers.Dense(64, activation="swish", kernel_initializer='he_normal', name="dense_2")(x)
x = layers.Dense(64, activation="swish", kernel_initializer='he_normal', name="dense_3")(x)
x = layers.Dense(64, activation="swish", kernel_initializer='he_normal', name="dense_4")(x)

# Output layer with identity function (None)
outputs = layers.Dense(1, activation=None, name="iv_output")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="rbergomi_ann")


#  Compile with RMSE Loss & Default Adam

batch_size = 256
epochs = 500  # "We allow for 500 epochs at most"

model.compile(
    optimizer=keras.optimizers.Adam(), # Keras default parameters
    loss=rmse_loss, # Using RMSE as loss function
    metrics=[
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.RootMeanSquaredError(name="rmse"),
        keras.metrics.MeanAbsoluteError(name="mae"),
        keras.metrics.MeanAbsolutePercentageError(name="mape") 
    ]
)


# Callbacks & Training

pytorch_logger = PyTorchStyleLogger(epochs)

# Reduce Learning Rate on Plateau
# Halves the LR if validation RMSE doesn't improve for 30 epochs, down to 1e-5
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_rmse",  # The metric to monitor
    factor=0.5,          # Halve the learning rate
    patience=30,         # Wait 30 epochs
    min_lr=1e-5,         # Do not drop below this value
    verbose=1            # Prints a message to the console when a drop occurs
)

# Early stopping patience 
# It must be higher than the reduce_lr patience so the model actually 
# has time to train at the new, lower learning rates before giving up entirely.
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=60,         
    restore_best_weights=True,
    verbose=1
)

print(f"Starting training for up to {epochs} epochs on mini-batches...\n")
print("-" * 75)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[pytorch_logger, reduce_lr, early_stopping], 
    verbose=0
)


# Evaluate and Save

print("\nTraining complete! Using BEST model weights for final evaluation...")
# Note: EarlyStopping automatically restores best weights due to `restore_best_weights=True`

eval_results = model.evaluate(X_val_scaled, y_val, verbose=0)
final_loss = eval_results[0]
final_mse = eval_results[1]
final_rmse = eval_results[2]
final_mae = eval_results[3]
final_mape = eval_results[4]

y_pred = model.predict(X_val_scaled, verbose=0)
ss_res = np.sum((y_val - y_pred) ** 2)
ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("-" * 30)
print("FINAL TEST SET PERFORMANCE")
print("-" * 30)
print(f"Loss (RMSE): {final_loss:.4e}")
print(f"MSE:   {final_mse:.4e}")
print(f"RMSE:  {final_rmse:.4e}")
print(f"MAE:   {final_mae:.4e}")
print(f"MAPE:  {final_mape:.4f}%")
print(f"R^2:   {r2:.6f}")
print("-" * 30)

model.save("rbergomi_ann.keras")
print("\nSaved model to 'rbergomi_ann.keras'")


# EXPORT PREDICTIONS TO CSV

print("\nGenerating CSV of predicted vs true Implied Volatilities...")

ann_iv = y_pred.flatten()
true_iv = y_val.flatten()

df_iv = pd.DataFrame({
    "ANN_IV": ann_iv,
    "True_IV": true_iv
})

csv_filename = "rbergomi_iv_errors.csv"
df_iv.to_csv(csv_filename, index=False)
print(f"Predictions saved successfully to: '{csv_filename}'")