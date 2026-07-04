#******** Necessary Imports **************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.decomposition import PCA
# from scipy.stats import f, norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

from utils import *


# Filepaths for data loading
project_path = Path(__file__).resolve().parent
normal_data_path = project_path / "data/normal.xlsx" 
hold_fault_data_path = project_path / "data/v102_fail_hold_fault.xlsx"
temp_fault_data_path = project_path / "data/temperature_fault.xlsx"
pres_fault_data_path = project_path / "data/pressure_fault.xlsx"



# Set the seed for all random number generators
set_global_seed(42) 


# Device configuration for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")


# Parameters
target_far = 0.01               # Allowable FAR on the calibration data (1% FAR, 99th Percentile)
epochs = 500                    # Maximum training epochs
patience = 10                   # Number of epochs to consider for convergence (When there is no improvement in the validation loss)
min_delta = 1e-4                # Convergence criteria, |f(w_t) - f(w_t-1) <= min_delta for 10 consecutive epochs
fault_start_time = (50*60) +1   # Time of fault injection in the faulty dataset (50 min, t starts from 0second)


#************ Data Loading & Preprocessing ***********************
df1 = pd.read_excel(normal_data_path) 

# Drop the first row of the DataFrame (it contains units, not data)
df1 = df1.iloc[1:, :]

# Convert all columns to numeric numpy array of floats
data = df1.values.astype(float) 

T = data.shape[0]

# first 80% for training, last 20% for validation
train_size = int(0.97 * T)
X_train = data[:train_size, :]
X_val = data[train_size:, :]


# # Remove constant columns based on training data only
# train_std = X_train.std(axis=0, ddof=0)
# keep_mask = train_std > 1e-12

# X_train = X_train[:, keep_mask]
# X_val   = X_val[:, keep_mask]

print("Kept features:", X_train.shape[1])


#************** Feature Scaling (Standardization) ***************
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)



#************* Autoencoder **********************
seed_ae = 42
set_global_seed(seed_ae)

print("\n========== Autoencoder ===========")
# Model parameters
input_dim = X_train_scaled.shape[1]
hidden1 = 160
hidden2 = 32
latent_dim = 12
lr = 0.0024869083266652416
batch_size = 32
weight_decay = 1.019762838023129e-06

# Dataloaders
train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))
train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed_ae)
val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed_ae)

# Autoencoder Model
model_ae = Autoencoder(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2, latent_dim=latent_dim, activation="Mish")
model_ae.to(device)
optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=lr, weight_decay=weight_decay)

# Training
print("Training...")
model_ae, train_history, val_history = train_step(model=model_ae, train_loader=train_loader, val_loader=val_loader,
                                                  optimizer=optimizer_ae, device=device,
                                                  epochs=epochs, patience=patience, min_delta=min_delta
                                                  )


epoch = np.arange(1, len(train_history) + 1)
plt.figure(figsize=(8, 4))
plt.plot(epoch, train_history, label='Train')
plt.plot(epoch, val_history, linestyle='--', label="Validation")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AE-Training')
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()





# #***************** PCA *****************************
num_pcs = 7
pca = PCA(n_components=num_pcs)
T_train = pca.fit_transform(X_train_scaled)     # scores
T_val = pca.transform(X_val_scaled)           # scores for validation data

P = pca.components_.T                           # loadings, shape (m, A)
lam = pca.explained_variance_                   # eigenvalues, shape (A,)

# Hotelling's T^2
T2_train = np.sum((T_train ** 2) / lam, axis=1)
T2_val = np.sum((T_val ** 2) / lam, axis=1)
alpha = 0.99
n = X_train.shape[0]

# T2_limit = (num_pcs * (n - 1) * (n + 1)) / (n * (n - num_pcs)) * f.ppf(alpha, num_pcs, n - num_pcs)
T2_limit = threshold_from_far(pre_residuals=T2_val, target_far=target_far)

print("\n========== PCA ===========")
print("T2 limit:", T2_limit)


# Reconstruction
X_train_hat = T_train @ P.T
X_val_hat = T_val @ P.T

# Residuals
E_train = X_train_scaled - X_train_hat
E_val = X_val_scaled - X_val_hat

# SPE (also called Q statistic)
SPE_train = np.sum(E_train ** 2, axis=1)
SPE_val = np.sum(E_val ** 2, axis=1)

# full PCA to get all eigenvalues
# pca_full = PCA().fit(X_train_scaled)
# all_lam = pca_full.explained_variance_

# discarded = all_lam[num_pcs:]

# theta1 = np.sum(discarded)
# theta2 = np.sum(discarded ** 2)
# theta3 = np.sum(discarded ** 3)

# z_alpha = norm.ppf(alpha)
# h0 = 1 - (2 * theta1 * theta3) / (3 * theta2 ** 2)

# SPE_limit = theta1 * (
#     (z_alpha * np.sqrt(2 * theta2 * h0 ** 2) / theta1)
#     + 1
#     + (theta2 * h0 * (h0 - 1) / (theta1 ** 2))
#     ) ** (1 / h0)

SPE_limit = threshold_from_far(pre_residuals=SPE_val, target_far=target_far)

print("SPE limit:", SPE_limit)



# #************ Threshold Analysis ***************************
model_ae.eval()
with torch.no_grad():

    # Reconstruction - AE
    recon_ae = model_ae(torch.tensor(X_val_scaled, dtype=torch.float32).to(device))
    recon_ae = recon_ae.cpu().numpy()
    recon_ae = scaler.inverse_transform(recon_ae)


# Residual & Threshold selection - AE
errors_ae = np.sqrt(np.mean((recon_ae - X_val)**2, axis= 1))
errors_ae = exponential_smoothing(errors_ae)
threshold_ae = threshold_from_far(pre_residuals=errors_ae, target_far=target_far)


print("\nFaultFree Calibration data:")
print("="*60)
labels = np.zeros(errors_ae.shape[0])
predicted_labels = (errors_ae > threshold_ae).astype(int)
accuarcy = accuracy_score(y_true=labels, y_pred=predicted_labels)
print(f"Accuracy on Validation - AE = {accuarcy*100:.2f}%")



labels = np.zeros(T2_val.shape[0])
predicted_labels = (T2_val > T2_limit).astype(int)
accuarcy = accuracy_score(y_true=labels, y_pred=predicted_labels)
print(f"Accuracy on Validation - PCA (T2) = {accuarcy*100:.2f}%")

labels = np.zeros(SPE_val.shape[0])
predicted_labels = (SPE_val > SPE_limit).astype(int)
accuarcy = accuracy_score(y_true=labels, y_pred=predicted_labels)
print(f"Accuracy on Validation - PCA (SPE) = {accuarcy*100:.2f}%")
print("="*60)




# #************* V-102 Fail Hold Fault Detection ****************************
df1 = pd.read_excel(hold_fault_data_path) 

# Drop the first row of the DataFrame (it contains units, not data)
df1 = df1.iloc[1:, :]

X_fault = df1.values.astype(float)

# X_fault = X_fault[:, keep_mask]

# Feature scaling
X_fault_scaled = scaler.transform(X_fault)

# Local threshold
X_prefault = X_fault[:fault_start_time]
X_prefault_scaled = scaler.transform(X_prefault)

with torch.no_grad():
    recon_prefault = model_ae(torch.tensor(X_prefault_scaled, dtype=torch.float32).to(device))
    recon_prefault = recon_prefault.cpu().numpy()
    recon_prefault = scaler.inverse_transform(recon_prefault)

prefault_err = np.sqrt(np.mean((recon_prefault - X_prefault)**2, axis=1))
prefault_err = exponential_smoothing(prefault_err)

threshold_ae_local = np.quantile(prefault_err, 0.99)



model_ae.eval()
with torch.no_grad():
    # Reconstruction - AE
    recon_ae = model_ae(torch.tensor(X_fault_scaled, dtype=torch.float32).to(device))
    recon_ae = recon_ae.cpu().numpy()
    recon_ae = scaler.inverse_transform(recon_ae)


# Residuals - AE
faulty_residuals_ae = np.sqrt(np.mean((recon_ae - X_fault)**2, axis=1))
faulty_residuals_ae = exponential_smoothing(faulty_residuals_ae)

# Predicted labels
predicted_labels_ae = (faulty_residuals_ae > threshold_ae_local).astype(int)

# True labels
n_samples = X_fault.shape[0]
fault_start_idx = fault_start_time
fault_labels = np.zeros(n_samples)
fault_labels[fault_start_idx:] = 1

# Metrics - AE
fdr_ae, far_ae, acc_ae = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_ae)
dd_ae, ttd_ae = get_detection_delay(predicted_labels_ae, fault_start_idx, ts=1)


# PCA Hotelling's T^2
T_prefault = pca.transform(X_prefault_scaled)
T2_prefault = np.sum((T_prefault**2)/lam, axis=1)

X_prefault_hat = T_prefault @ P.T
E_prefault = X_prefault_scaled - X_prefault_hat
SPE_prefault = np.sum(E_prefault**2, axis=1)

T2_limit_local = np.quantile(T2_prefault, 0.99)
SPE_limit_local = np.quantile(SPE_prefault, 0.99)


T_fault = pca.transform(X_fault_scaled)
T2_fault = np.sum((T_fault ** 2) / lam, axis=1)
X_fault_hat = T_fault @ P.T
E_fault = X_fault_scaled - X_fault_hat
SPE_fault = np.sum(E_fault ** 2, axis=1)
predicted_labels_T2 = (T2_fault > T2_limit_local).astype(int)
predicted_labels_SPE = (SPE_fault > SPE_limit_local).astype(int)


# Metrics - PCA T^2
fdr_T2, far_T2, acc_T2 = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_T2)
dd_T2, ttd_T2 = get_detection_delay(predicted_labels_T2, fault_start_idx, ts=1)

# Metrics - PCA SPE
fdr_SPE, far_SPE, acc_SPE = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_SPE)
dd_SPE, ttd_SPE = get_detection_delay(predicted_labels_SPE, fault_start_idx, ts=1)



#********* Compute mean ± std ***************
print("\n===== 1. V-102 Fail Hold Fault Detection =====")
print("=" * 55)
print(f"{'Metric':<12}{'AE':>12}{'PCA (T2)':>15}{'PCA (SPE)':>15}")
print("-" * 55)

print(f"{'FDR (%)':<12}{fdr_ae*100:>12.2f}{fdr_T2*100:>15.2f}{fdr_SPE*100:>15.2f}")
print(f"{'FAR (%)':<12}{far_ae*100:>12.2f}{far_T2*100:>15.2f}{far_SPE*100:>15.2f}")
print(f"{'ACC (%)':<12}{acc_ae*100:>12.2f}{acc_T2*100:>15.2f}{acc_SPE*100:>15.2f}")
print(f"{'TTD (min)':<12}{ttd_ae/60:>12.2f}{ttd_T2/60:>15.2f}{ttd_SPE/60:>15.2f}")

print("=" * 55)



# Plotting
t_vector = np.arange(X_fault.shape[0]) 
plt.figure(figsize=(8, 4))
plt.plot(t_vector, faulty_residuals_ae, label='AE', color='blue')
plt.axhline(y=threshold_ae_local, color='red', linestyle='--', label='Threshold')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("V-102 Fail Hold Fault Detection - AE Residuals")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()


# Plot T^2 and SPE with limits
plt.figure(figsize=(8, 4))
plt.plot(t_vector,T2_fault, label='T²')
plt.axhline(T2_limit_local, color='red', linestyle='--', label='T² Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("V-102 Fail Hold Fault Detection - Hotelling's T²")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()



plt.figure(figsize=(8, 4))
plt.plot(t_vector, SPE_fault, label='SPE')
plt.axhline(SPE_limit_local, color='red', linestyle='--', label='SPE Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("V-102 Fail Hold Fault Detection - SPE (Q statistic)")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()





# #************* Temperature Fault Detection ****************************
df2 = pd.read_excel(temp_fault_data_path) 

# Drop the first row of the DataFrame (it contains units, not data)
df2 = df2.iloc[1:, :]

X_fault = df2.values.astype(float)

# X_fault = X_fault[:, keep_mask]


X_prefault = X_fault[:fault_start_idx]
X_prefault_scaled = scaler.transform(X_prefault)

with torch.no_grad():
    recon_prefault = model_ae(torch.tensor(X_prefault_scaled, dtype=torch.float32).to(device))
    recon_prefault = recon_prefault.cpu().numpy()
    recon_prefault = scaler.inverse_transform(recon_prefault)

prefault_err = np.sqrt(np.mean((recon_prefault - X_prefault)**2, axis=1))
prefault_err = exponential_smoothing(prefault_err)

threshold_ae_local = np.quantile(prefault_err, 0.99)


# Feature scaling
X_fault_scaled = scaler.transform(X_fault)

model_ae.eval()
with torch.no_grad():
    # Reconstruction - AE
    recon_ae = model_ae(torch.tensor(X_fault_scaled, dtype=torch.float32).to(device))
    recon_ae = recon_ae.cpu().numpy()
    recon_ae = scaler.inverse_transform(recon_ae)


# Residuals - AE
faulty_residuals_ae = np.sqrt(np.mean((recon_ae - X_fault)**2, axis=1))
faulty_residuals_ae = exponential_smoothing(faulty_residuals_ae)

# Predicted labels
predicted_labels_ae = (faulty_residuals_ae > threshold_ae_local).astype(int)

# True labels
n_samples = X_fault.shape[0]
fault_start_idx = fault_start_time
fault_labels = np.zeros(n_samples)
fault_labels[fault_start_idx:] = 1

# Metrics - AE
fdr_ae, far_ae, acc_ae = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_ae)
dd_ae, ttd_ae = get_detection_delay(predicted_labels_ae, fault_start_idx, ts=1)


# PCA Hotelling's T^2
T_prefault = pca.transform(X_prefault_scaled)
T2_prefault = np.sum((T_prefault**2)/lam, axis=1)

X_prefault_hat = T_prefault @ P.T
E_prefault = X_prefault_scaled - X_prefault_hat
SPE_prefault = np.sum(E_prefault**2, axis=1)

T2_limit_local = np.quantile(T2_prefault, 0.99)
SPE_limit_local = np.quantile(SPE_prefault, 0.99)


T_fault = pca.transform(X_fault_scaled)
T2_fault = np.sum((T_fault ** 2) / lam, axis=1)
X_fault_hat = T_fault @ P.T
E_fault = X_fault_scaled - X_fault_hat
SPE_fault = np.sum(E_fault ** 2, axis=1)
predicted_labels_T2 = (T2_fault > T2_limit_local).astype(int)
predicted_labels_SPE = (SPE_fault > SPE_limit_local).astype(int)


# Metrics - PCA T^2
fdr_T2, far_T2, acc_T2 = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_T2)
dd_T2, ttd_T2 = get_detection_delay(predicted_labels_T2, fault_start_idx, ts=1)

# Metrics - PCA SPE
fdr_SPE, far_SPE, acc_SPE = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_SPE)
dd_SPE, ttd_SPE = get_detection_delay(predicted_labels_SPE, fault_start_idx, ts=1)



#********* Compute mean ± std ***************
print("\n===== 2. Temperature Fault Detection =====")
print("=" * 55)
print(f"{'Metric':<12}{'AE':>12}{'PCA (T2)':>15}{'PCA (SPE)':>15}")
print("-" * 55)

print(f"{'FDR (%)':<12}{fdr_ae*100:>12.2f}{fdr_T2*100:>15.2f}{fdr_SPE*100:>15.2f}")
print(f"{'FAR (%)':<12}{far_ae*100:>12.2f}{far_T2*100:>15.2f}{far_SPE*100:>15.2f}")
print(f"{'ACC (%)':<12}{acc_ae*100:>12.2f}{acc_T2*100:>15.2f}{acc_SPE*100:>15.2f}")
print(f"{'TTD (min)':<12}{ttd_ae/60:>12.2f}{ttd_T2/60:>15.2f}{ttd_SPE/60:>15.2f}")

print("=" * 55)



# Plotting
t_vector = np.arange(X_fault.shape[0]) 
plt.figure(figsize=(8, 4))
plt.plot(t_vector, faulty_residuals_ae, label='AE', color='blue')
plt.axhline(y=threshold_ae_local, color='red', linestyle='--', label='Threshold')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Temperature Fault Detection - AE Residuals")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

# Plot T^2 and SPE with limits
plt.figure(figsize=(8, 4))
plt.plot(t_vector,T2_fault, label='T²')
plt.axhline(T2_limit_local, color='red', linestyle='--', label='T² Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Temperature Fault Detection - Hotelling's T²")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()



plt.figure(figsize=(8, 4))
plt.plot(t_vector, SPE_fault, label='SPE')
plt.axhline(SPE_limit_local, color='red', linestyle='--', label='SPE Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Temperature Fault Detection - SPE (Q statistic)")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()




# #************* Pressure Fault Detection ****************************
df3 = pd.read_excel(pres_fault_data_path) 

# Drop the first row of the DataFrame (it contains units, not data)
df3 = df3.iloc[1:, :]

X_fault = df3.values.astype(float)

# X_fault = X_fault[:, keep_mask]


X_prefault = X_fault[:fault_start_idx]
X_prefault_scaled = scaler.transform(X_prefault)

with torch.no_grad():
    recon_prefault = model_ae(torch.tensor(X_prefault_scaled, dtype=torch.float32).to(device))
    recon_prefault = recon_prefault.cpu().numpy()
    recon_prefault = scaler.inverse_transform(recon_prefault)

prefault_err = np.sqrt(np.mean((recon_prefault - X_prefault)**2, axis=1))
prefault_err = exponential_smoothing(prefault_err)

threshold_ae_local = np.quantile(prefault_err, 0.99)


# Feature scaling
X_fault_scaled = scaler.transform(X_fault)

model_ae.eval()
with torch.no_grad():
    # Reconstruction - AE
    recon_ae = model_ae(torch.tensor(X_fault_scaled, dtype=torch.float32).to(device))
    recon_ae = recon_ae.cpu().numpy()
    recon_ae = scaler.inverse_transform(recon_ae)


# Residuals - AE
faulty_residuals_ae = np.sqrt(np.mean((recon_ae - X_fault)**2, axis=1))
faulty_residuals_ae = exponential_smoothing(faulty_residuals_ae)

# Predicted labels
predicted_labels_ae = (faulty_residuals_ae > threshold_ae_local).astype(int)

# True labels
n_samples = X_fault.shape[0]
fault_start_idx = fault_start_time
fault_labels = np.zeros(n_samples)
fault_labels[fault_start_idx:] = 1

# Metrics - AE
fdr_ae, far_ae, acc_ae = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_ae)
dd_ae, ttd_ae = get_detection_delay(predicted_labels_ae, fault_start_idx, ts=1)


# PCA Hotelling's T^2
T_prefault = pca.transform(X_prefault_scaled)
T2_prefault = np.sum((T_prefault**2)/lam, axis=1)

X_prefault_hat = T_prefault @ P.T
E_prefault = X_prefault_scaled - X_prefault_hat
SPE_prefault = np.sum(E_prefault**2, axis=1)

T2_limit_local = np.quantile(T2_prefault, 0.99)
SPE_limit_local = np.quantile(SPE_prefault, 0.99)


T_fault = pca.transform(X_fault_scaled)
T2_fault = np.sum((T_fault ** 2) / lam, axis=1)
X_fault_hat = T_fault @ P.T
E_fault = X_fault_scaled - X_fault_hat
SPE_fault = np.sum(E_fault ** 2, axis=1)
predicted_labels_T2 = (T2_fault > T2_limit_local).astype(int)
predicted_labels_SPE = (SPE_fault > SPE_limit_local).astype(int)


# Metrics - PCA T^2
fdr_T2, far_T2, acc_T2 = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_T2)
dd_T2, ttd_T2 = get_detection_delay(predicted_labels_T2, fault_start_idx, ts=1)

# Metrics - PCA SPE
fdr_SPE, far_SPE, acc_SPE = compute_fdr_far_mdr(true_labels=fault_labels, pred_labels=predicted_labels_SPE)
dd_SPE, ttd_SPE = get_detection_delay(predicted_labels_SPE, fault_start_idx, ts=1)



#********* Compute mean ± std ***************
print("\n===== 3. Pressure Fault Detection =====")
print("=" * 55)
print(f"{'Metric':<12}{'AE':>12}{'PCA (T2)':>15}{'PCA (SPE)':>15}")
print("-" * 55)

print(f"{'FDR (%)':<12}{fdr_ae*100:>12.2f}{fdr_T2*100:>15.2f}{fdr_SPE*100:>15.2f}")
print(f"{'FAR (%)':<12}{far_ae*100:>12.2f}{far_T2*100:>15.2f}{far_SPE*100:>15.2f}")
print(f"{'ACC (%)':<12}{acc_ae*100:>12.2f}{acc_T2*100:>15.2f}{acc_SPE*100:>15.2f}")
print(f"{'TTD (min)':<12}{ttd_ae/60:>12.2f}{ttd_T2/60:>15.2f}{ttd_SPE/60:>15.2f}")

print("=" * 55)



# Plotting
t_vector = np.arange(X_fault.shape[0]) 
plt.figure(figsize=(8, 4))
plt.plot(t_vector, faulty_residuals_ae, label='AE', color='blue')
plt.axhline(y=threshold_ae_local, color='red', linestyle='--', label='Threshold')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Pressure Fault Detection - AE Residuals")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()

# Plot T^2 and SPE with limits
plt.figure(figsize=(8, 4))
plt.plot(t_vector,T2_fault, label='T²')
plt.axhline(T2_limit_local, color='red', linestyle='--', label='T² Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Pressure Fault Detection - Hotelling's T²")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()



plt.figure(figsize=(8, 4))
plt.plot(t_vector, SPE_fault, label='SPE')
plt.axhline(SPE_limit_local, color='red', linestyle='--', label='SPE Limit')
plt.axvline(x=fault_start_time, color='orange', linestyle='--', label='Fault Start')
plt.title("Pressure Fault Detection - SPE (Q statistic)")
plt.xlabel('Time [s]')
plt.ylabel('RMS Residual (log scale)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()