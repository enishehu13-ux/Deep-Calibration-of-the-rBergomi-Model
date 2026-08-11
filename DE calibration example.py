import os
# 1. Turn off the oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 2. Turn off TensorFlow's C++ informational and warning logs (0=all, 1=no INFO, 2=no WARNINGS, 3=no ERRORS)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import joblib
import keras
from scipy.optimize import differential_evolution


# ANN PREDICTION (Numpy Forward Pass)


def ann_predict_numpy(X_input, weights):
    """
    Manual forward pass through the rBergomi ANN using the Swish activation function.
    Assumes 4 hidden layers of 64 neurons.
    """
    def swish(z):
        return z * (1.0 / (1.0 + np.exp(-z)))

    x = X_input
    for i in range(0, 8, 2):
        x = swish(np.dot(x, weights[i]) + weights[i+1])
    output = np.dot(x, weights[8]) + weights[9]
    return output.flatten()



# DIFFERENTIAL EVOLUTION CALIBRATION


def calibrate_rbergomi_3param_de(
    market_ivs, strikes, t_years, fixed_xi_curve,
    model_path="rbergomi_ann.keras", scaler_path="rbergomi_scaler.pkl"
):
    print("Loading rBergomi ANN model and scaler...")
    try:
        model = keras.models.load_model(model_path, compile=False)
        weights = model.get_weights()
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        raise SystemExit(1)

    num_points = len(market_ivs)

    def objective_function(params):
        alpha = params[0]
        rho   = params[1]
        eta   = params[2]

        X_raw = np.zeros((num_points, 13))
        X_raw[:, 0] = alpha
        X_raw[:, 1] = rho
        X_raw[:, 2] = eta
        
        for i in range(8):
            X_raw[:, 3+i] = fixed_xi_curve[i]
            
        X_raw[:, 11] = t_years
        X_raw[:, 12] = strikes

        X_scaled = scaler.transform(X_raw)
        predicted_ivs = ann_predict_numpy(X_scaled, weights)

        # We minimize RMSE
        rmse = np.sqrt(np.mean((predicted_ivs - market_ivs)**2))
        return rmse

    bounds = [
        (-0.475, 0.0),   # alpha
        (-0.95, -0.10),  # rho
        (0.5, 4.0),      # eta
    ]

    print("Running Differential Evolution Optimization (Optimizing H, rho, eta)...")
    res = differential_evolution(
        func=objective_function,
        bounds=bounds,
        strategy='best1bin',
        popsize=45,        
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,       # Runs L-BFGS-B at the end for extreme precision
        disp=True
    )

    final_rmse = res.fun

    if res.success:
        # Calculate MAE to see if outliers are skewing the RMSE
        alpha, rho, eta = res.x
        X_raw = np.zeros((num_points, 13))
        X_raw[:, 0], X_raw[:, 1], X_raw[:, 2] = alpha, rho, eta
        for i in range(8): X_raw[:, 3+i] = fixed_xi_curve[i]
        X_raw[:, 11], X_raw[:, 12] = t_years, strikes
        
        predicted_ivs = ann_predict_numpy(scaler.transform(X_raw), weights)
        final_mae = np.mean(np.abs(predicted_ivs - market_ivs))
        
        return True, res.x, final_rmse, final_mae
    else:
        return False, res.message, final_rmse, None



# EXECUTION EXAMPLE


if __name__ == "__main__":
    
    # 1. LOAD THE TARGET SURFACE
    csv_file = "rbergomi_66K.csv"
    print(f"Loading target surface from {csv_file}...")
    
    df = pd.read_csv(csv_file)
        
    clean_t_years = df['T'].values
    clean_strikes = df['Strike'].values
    clean_mkt_ivs = df['IV'].values
    
    fixed_xi_curve = df[[f'xi_pillar_{i+1}' for i in range(8)]].iloc[0].values
    
    true_alpha = df['alpha'].iloc[0]
    true_rho   = df['rho'].iloc[0]
    true_eta   = df['eta'].iloc[0]
    true_H     = true_alpha + 0.5 

    print(f"Loaded {len(clean_mkt_ivs)} valid target IVs.")

    # 2. RUN THE ANN CALIBRATION
    success, result, final_rmse, final_mae = calibrate_rbergomi_3param_de(
        market_ivs=clean_mkt_ivs,
        strikes=clean_strikes,
        t_years=clean_t_years,
        fixed_xi_curve=fixed_xi_curve,
        model_path="rbergomi_ann.keras",     
        scaler_path="rbergomi_scaler.pkl"      
    )

    if success:
        alpha_opt = result[0]
        rho_opt   = result[1]
        eta_opt   = result[2]
        H_opt     = alpha_opt + 0.5 
        
        print("\n=== rBERGOMI CALIBRATION RESULTS ===")
        print("--------------------------------------------------")
        print(f"          Target (Truth)      | Calibrated via ANN (DE)")
        print("--------------------------------------------------")
        print(f"H:       {true_H: .4f}             | {H_opt: .4f}")
        print(f"Rho:     {true_rho: .4f}             | {rho_opt: .4f}")
        print(f"Eta:     {true_eta: .4f}             | {eta_opt: .4f}")
        print("--------------------------------------------------")
        print(f"Final Root Mean Squared Error (RMSE): {final_rmse:.8e}")
        print(f"Final Mean Absolute Error (MAE):      {final_mae:.8e}")
        print("--------------------------------------------------")
        
    else:
        print("Calibration failed:", result)
        print("Final RMSE before failing:", final_rmse)