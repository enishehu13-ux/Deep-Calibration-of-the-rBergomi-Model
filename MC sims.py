import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.stats import qmc
import csv


# MATH & RBERGOMI ENGINE SETUP

def g(x, a): return x**a
def b(k, a): return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    cov_mat = np.array([[0.,0.],[0.,0.]])
    cov_mat[0,0] = 1./n
    cov_mat[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov_mat[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov_mat[1,0] = cov_mat[0,1]
    return cov_mat

def bs(F, K, V, o = 'call'):
    w = 1 if o == 'call' else -1 if o == 'put' else 2 * (K > 1.0) - 1
    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bsinv(P, F, K, t, o = 'call'):
    w = 1 if o == 'call' else -1 if o == 'put' else 2 * (K > 1.0) - 1
    P = np.maximum(P, np.maximum(w * (F - K), 0))
    def error(s): return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s

vec_bsinv = np.vectorize(bsinv)

class rBergomi(object):
    def __init__(self, n=100, N=1000, T=1.00, a=-0.4, rho=-0.9):
        self.T = T 
        self.n = n 
        self.dt = 1.0/self.n 
        self.s = int(self.n * self.T) 
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:] 
        self.a = a 
        self.N = N 
        self.rho = rho
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        # 1. Initialize the random generator
        rng = np.random.multivariate_normal
        
        # 2. Draw ONLY HALF the required paths (N // 2)
        half_dW = rng(self.e, self.c, (self.N // 2, self.s))
        
        # 3. Create the antithetic counterparts (-Z)
        anti_dW = -half_dW
        
        # 4. Stack them vertically to return exactly N paths
        return np.concatenate((half_dW, anti_dW), axis=0)

    def Y(self, dW):
        Y1 = np.zeros((self.N, 1 + self.s)) 
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:,i] = dW[:,i-1,1] 
        G = np.zeros(1 + self.s) 
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a)/self.n, self.a)
        X = dW[:,:,0] 
        GX = fftconvolve(X, G[np.newaxis, :], axes=1)
        Y2 = GX[:,:1 + self.s]
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self):
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0):
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi=1.0, eta=1.0):
        self.xi = xi
        self.eta = eta
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * self.t**(2 * self.a + 1))
        return V

    def S(self, V, dB, S0=1):
        self.S0 = S0
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * self.dt
        integral = np.cumsum(increments, axis=1)
        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S

    def S1(self, V, dW1, rho, S0=1):
        increments = rho * np.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * self.dt
        integral = np.cumsum(increments, axis=1)
        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S


# USER SAMPLING FUNCTIONS

T_INTERVALS = np.array([
    [0.003, 0.030], [0.030, 0.090], [0.090, 0.150], [0.150, 0.300],
    [0.300, 0.500], [0.500, 0.750], [0.750, 1.000], [1.000, 1.250],
    [1.250, 1.500], [1.500, 2.000], [2.000, 2.500],
])

def sample_expiries_years(rng):
    T_years = np.zeros(len(T_INTERVALS))
    for j, (a, b) in enumerate(T_INTERVALS):
        T_years[j] = rng.uniform(a, b)
    return T_years

def strikes_for_T(S0, T_years, rng, m=13, l=0.55, u=0.30):
    sqrt_T = np.sqrt(T_years)
    K_min = S0 * (1.0 - l * sqrt_T)
    K_max = S0 * (1.0 + u * sqrt_T)
    left_upper = S0 * (1.0 - 0.20 * sqrt_T)
    left_strikes = rng.uniform(K_min, left_upper, 4)
    center_lower = left_upper
    center_upper = S0 * (1.0 + 0.20 * sqrt_T)
    center_strikes = rng.uniform(center_lower, center_upper, 7)
    right_lower = center_upper
    right_upper = K_max
    right_strikes = rng.uniform(right_lower, right_upper, 2)
    Ks = np.concatenate([left_strikes, center_strikes, right_strikes])
    return np.sort(Ks)


# LHS SAMPLING & DATA STORAGE SETUP

num_samples = 10 
rng = np.random.default_rng(42)

sampler = qmc.LatinHypercube(d=11, seed=42)
lhs_samples = sampler.random(n=num_samples)

l_bounds = [0.01] * 8 + [0.5, -0.95, 0.025]
u_bounds = [0.16] * 8 + [4.0, -0.10, 0.500]
scaled_samples = qmc.scale(lhs_samples, l_bounds, u_bounds)

xi_samples  = scaled_samples[:, 0:8]
eta_samples = scaled_samples[:, 8]
rho_samples = scaled_samples[:, 9]
H_samples   = scaled_samples[:, 10]
alpha_samples = H_samples - 0.5

curve_times = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.5])

# Initialize CSV File and Headers
csv_filename = "rbergomi_dataset.csv"
headers = (
    ['alpha', 'rho', 'eta'] + 
    [f'xi_pillar_{i+1}' for i in range(8)] + 
    ['T', 'Strike', 'IV']
)
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# Initialize RAM Lists for .npz
X_data = []
y_data = []


#  MAIN DATASET GENERATION LOOP

for i in range(num_samples):
    current_xi_pillars = xi_samples[i]
    current_eta        = eta_samples[i]
    current_rho        = rho_samples[i]
    current_alpha      = alpha_samples[i]
    
    print(f"\n--- Processing LHS Sample {i+1}/{num_samples} ---")
    
    sampled_Ts = sample_expiries_years(rng)
    
    for target_T in sampled_Ts:
        actual_T = max(target_T, 0.001)
        rB = rBergomi(n=312*4, N=500, T=actual_T, a=current_alpha, rho=current_rho)
        
        indices = np.digitize(rB.t[0, :], curve_times)
        indices = np.clip(indices, 0, len(current_xi_pillars) - 1)
        xi_curve = current_xi_pillars[indices][np.newaxis, :]
        
        dW1 = rB.dW1()
        Ya = rB.Y(dW1)
        V = rB.V(Ya, xi=xi_curve, eta=current_eta) 
        S1 = rB.S1(V, dW1, rho=current_rho)
        
        Ks = strikes_for_T(S0=1.0, T_years=actual_T, rng=rng)
        K_matrix = Ks[np.newaxis, :] 
        
        S1T = S1[:, -1][:, np.newaxis]
        QV = np.sum(V, axis=1)[:, np.newaxis] * rB.dt
        Q = np.max(QV) + 1e-9 
        
        X_price = bs(S1T, K_matrix, (1 - current_rho**2) * QV)
        Y_price = bs(S1T, K_matrix, current_rho**2 * (Q - QV))
        eY_price = bs(1., K_matrix, current_rho**2 * Q)
        
        c = np.zeros_like(Ks)[np.newaxis, :]
        for j in range(len(Ks)):
            cov_mat = np.cov(X_price[:, j], Y_price[:, j])
            c[0, j] = -cov_mat[0, 1] / cov_mat[1, 1]
            
        mixed_payoffs = X_price + c * (Y_price - eY_price)
        mixed_prices = np.mean(mixed_payoffs, axis=0)[:, np.newaxis]
        
        mixed_vols = vec_bsinv(mixed_prices, 1., np.transpose(K_matrix), actual_T)
        xi_list = current_xi_pillars.tolist()
        
        # Open CSV in Append Mode for this specific maturity
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Extract and save all 13 strikes
            for j in range(len(Ks)):
                strike_val = Ks[j]
                iv_val = mixed_vols[j, 0]
                
                # Create the standard Python list row
                row_data = [current_alpha, current_rho, current_eta] + xi_list + [actual_T, strike_val]
                
                # 1. Save to Hard Drive (.csv)
                writer.writerow(row_data + [iv_val])
                
                # 2. Save to RAM (.npz)
                X_data.append(row_data)
                y_data.append(iv_val)
                
        print(f"Expiry: {actual_T:5.3f}Y | Steps Simulated: {rB.s:4d} | Appended 13 Rows to CSV/RAM")


# SAVE TO .NPZ COMPRESSED ARCHIVE

print("\n--- Compressing RAM to .npz ---")

# Convert the master lists to highly optimized Deep Learning arrays
X_array = np.array(X_data, dtype=np.float32)
y_array = np.array(y_data, dtype=np.float32).reshape(-1, 1)

npz_filename = "rbergomi_dataset.npz"
np.savez_compressed(npz_filename, X=X_array, y=y_array)

print(f"Successfully finalized {csv_filename} and {npz_filename}!")
print(f"Total Rows Generated: {len(X_data)}")
print(f"Dataset Shape X (Inputs):  {X_array.shape}")
print(f"Dataset Shape y (Outputs): {y_array.shape}")