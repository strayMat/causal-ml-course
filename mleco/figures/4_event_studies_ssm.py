# %% 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from mleco.figures.style_figs import *
# %%
# generate a AR(1) process with statsmodel
np.random.seed(12345)
ar_params = np.array([1, -0.8, 0.4])  # AR(1) with phi = 0.5
ma_params = np.array([1])       # No MA terms

# Create ARMA process
ar1_process = ArmaProcess(ar_params, ma_params)

# Generate artificial data
n_samples = 200  # Number of samples
np.random.seed(42)  # For reproducibility
simulated_data = ar1_process.generate_sample(nsample=n_samples)
 
# Plot the data
time = np.arange(len(simulated_data))
plt.figure(figsize=(10, 6))
plt.plot(time, simulated_data, label='Simulated AR(1) Data')
plt.title('Simulated AR(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Simulated an MA(1) process
ar_params = np.array([1])       # No AR terms
ma_params = np.array([1, 0.7, 4])  # MA(1) with theta = 0.7

# Create ARMA process
ma1_process = ArmaProcess(ar_params, ma_params)
# Generate artificial data
n_samples = 200  # Number of samples
simulated_data = ma1_process.generate_sample(nsample=n_samples)
# Plot the data
time = np.arange(len(simulated_data))
plt.figure(figsize=(10, 6))
plt.plot(time, simulated_data, label='Simulated MA(1) Data')
plt.title('Simulated MA(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
# %%
# simulate a ARMA(p,q) process
# Simulated an MA(1) process
ar_params = np.array([1, 0.5])       # No AR terms
ma_params = np.array([1, 0.7, 4])  # MA(1) with theta = 0.7

# Create ARMA process
ma1_process = ArmaProcess(ar_params, ma_params)
# Generate artificial data
n_samples = 200  # Number of samples
simulated_data = ma1_process.generate_sample(nsample=n_samples)
# Plot the data
time = np.arange(len(simulated_data))
plt.figure(figsize=(10, 6))
plt.plot(time, simulated_data, label='Simulated MA(1) Data')
plt.title('Simulated MA(1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
# %% 