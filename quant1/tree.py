import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Download data for BTC, ETH, and SPY
assets = ['BTC-USD', 'ETH-USD', 'SPY']
data = yf.download(assets, start='2010-01-01', end='2024-10-10')['Adj Close']

# Check if data is downloaded correctly
if data is None or data.empty:
    print("Data download failed. Please check your internet connection or the asset symbols.")
    exit()

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate rolling Sharpe ratios
window = 252  # 1-year rolling window

def rolling_sharpe_ratio(returns, window):
    rolling_mean = returns.rolling(window).mean() * np.sqrt(252)
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return rolling_mean / rolling_std

rolling_sharpe = rolling_sharpe_ratio(returns, window)

# Calculate rolling correlations
rolling_corr_btc_eth = returns['BTC-USD'].rolling(window).corr(returns['ETH-USD'])
rolling_corr_btc_spy = returns['BTC-USD'].rolling(window).corr(returns['SPY'])
rolling_corr_eth_spy = returns['ETH-USD'].rolling(window).corr(returns['SPY'])

# Prepare the data for HMM
# Combine rolling Sharpe ratios and correlations
hmm_data = pd.DataFrame({
    'BTC_Sharpe': rolling_sharpe['BTC-USD'],
    'ETH_Sharpe': rolling_sharpe['ETH-USD'],
    'SPY_Sharpe': rolling_sharpe['SPY'],
    'BTC_ETH_Corr': rolling_corr_btc_eth,
    'BTC_SPY_Corr': rolling_corr_btc_spy,
    'ETH_SPY_Corr': rolling_corr_eth_spy
}).dropna()

# Reset index to ensure compatibility
hmm_data.reset_index(inplace=True)
hmm_data.rename(columns={'index': 'Date'}, inplace=True)

# Check for NaN values
if hmm_data.isnull().values.any():
    print("NaN values found in hmm_data. Dropping NaNs.")
    hmm_data.dropna(inplace=True)

# Ensure that hmm_data is not empty after dropping NaNs
if hmm_data.empty:
    print("hmm_data is empty after dropping NaNs. Cannot proceed.")
    exit()

# Scale the data
scaler = StandardScaler()
hmm_data_scaled = scaler.fit_transform(hmm_data.drop(columns=['Date']))

# Check if data is properly scaled
if hmm_data_scaled is None or len(hmm_data_scaled) == 0:
    print("Data scaling failed.")
    exit()

# Fit Hidden Markov Model
num_states = 3  # Number of hidden states (adjust as needed)
hmm_model = GaussianHMM(n_components=num_states, covariance_type="full", n_iter=1000)

try:
    hmm_model.fit(hmm_data_scaled)
    # Predict hidden states
    hidden_states = hmm_model.predict(hmm_data_scaled)
except Exception as e:
    print(f"Error during HMM fitting or prediction: {e}")
    exit()

# Add hidden states to the DataFrame
hmm_data['Hidden_State'] = hidden_states

# Plot heatmaps to show differences in price action
for state in range(num_states):
    state_data = hmm_data[hmm_data['Hidden_State'] == state].drop(columns=['Date', 'Hidden_State'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(state_data.corr(), annot=True, cmap='coolwarm')
    plt.title(f'Heatmap of Correlations in Hidden State {state}')
    plt.show()

# Plot rolling Sharpe ratios colored by hidden states
plt.figure(figsize=(14, 7))
for state in range(num_states):
    idx = hmm_data[hmm_data['Hidden_State'] == state].index
    plt.scatter(hmm_data.loc[idx, 'Date'], rolling_sharpe['BTC-USD'].iloc[idx], label=f'State {state}', s=10)
plt.title('Rolling Sharpe Ratio of BTC Colored by Hidden States')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show()

# Plot rolling correlations colored by hidden states
plt.figure(figsize=(14, 7))
for state in range(num_states):
    idx = hmm_data[hmm_data['Hidden_State'] == state].index
    plt.scatter(hmm_data.loc[idx, 'Date'], rolling_corr_btc_eth.iloc[idx], label=f'State {state}', s=10)
plt.title('Rolling Correlation between BTC and ETH Colored by Hidden States')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.legend()
plt.show()
