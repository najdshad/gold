import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def load_data(file_path):
    """
    Load the CSV data.
    Expected CSV columns: timestamp, open, high, low, close, volume.
    """
    data = pd.read_csv(file_path, parse_dates=['Gmt time'], index_col='Gmt time')
    data.sort_index(inplace=True)
    return data

def prepare_features(data):
    """
    Compute features for the HMM.
    Here, we compute the percentage return from one candle to the next.
    """
    data['returns'] = data['Close'].pct_change()
    data = data.dropna()
    # Reshape returns for HMM input (n_samples x n_features)
    X = data['returns'].values.reshape(-1, 1)
    return X, data

def fit_hmm(X, n_states=2):
    """
    Fit a Gaussian HMM to the returns.
    We use 2 states as a simple model for bearish vs bullish regimes.
    """
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=28)
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states

def analyze_states(data, hidden_states):
    """
    Analyze each state to determine its characteristic mean return.
    The state with the higher average return is considered bullish,
    while the lower is bearish. We then decide overall bias based on
    which state is more dominant.
    """
    data['state'] = hidden_states
    state_stats = {}
    for state in np.unique(hidden_states):
        mean_return = data[data['state'] == state]['returns'].mean()
        state_stats[state] = mean_return

    # Identify bullish (higher mean return) and bearish (lower mean return) states.
    bullish_state = max(state_stats, key=state_stats.get)
    bearish_state = min(state_stats, key=state_stats.get)
    
    # Count occurrences of each state
    state_counts = data['state'].value_counts()
    
    # Determine overall bias: if bullish state appears more, bias is bullish.
    if state_counts[bullish_state] > state_counts[bearish_state]:
        bias = "bullish"
    else:
        bias = "bearish"
        
    return bias, state_stats, state_counts

def risk_management_recommendations():
    """
    Print some recommendations for position sizing and risk management:
    - Use fixed fractional position sizing (risk only 1-2% per trade).
    - Determine stop-loss based on market volatility (e.g., using ATR).
    - Aim for a risk-to-reward ratio of at least 1:2 (i.e., risking 1 to gain 2).
    - Consider dynamic or trailing stops to secure profits.
    - Backtest and optimize these parameters on historical data.
    """
    recommendations = """
Position Sizing and Risk Management Recommendations:
------------------------------------------------------
1. Fixed Fractional Position Sizing:
   - Risk only a small percentage (e.g., 1-2%) of your capital per trade.
   - For example, if your account balance is $10,000 and you risk 1%, then
     your maximum loss per trade should be $100.

2. Stop-Loss Placement:
   - Use technical indicators such as ATR (Average True Range) to set stop-loss levels,
     ensuring that your stops are not too tight to avoid being prematurely stopped out.

3. Risk-to-Reward Ratio:
   - Target a minimum risk-to-reward ratio of 1:2. For instance, if you risk 50 pips,
     aim for a target of at least 100 pips.
   - This can be optimized by backtesting different stop-loss and take-profit levels.

4. Trailing Stops:
   - Use trailing stops to lock in profits as the market moves in your favor,
     adjusting your exit point dynamically.

5. Backtesting and Optimization:
   - Simulate your strategy over historical data and adjust parameters such as the stop-loss distance,
     take-profit levels, and position size to maximize profit while strictly managing risk.

Example position size calculation:
------------------------------------------------------
def calculate_position_size(account_balance, risk_percentage, stop_loss_distance, pip_value):
    risk_amount = account_balance * (risk_percentage / 100)
    # Position size in lots = risk_amount / (stop_loss_distance * pip_value)
    position_size = risk_amount / (stop_loss_distance * pip_value)
    return position_size

By combining these techniques with robust backtesting, you can tailor your strategy to optimize for maximum profit given your risk tolerance.
    """
    print(recommendations)

def main():
    # Replace with the path to your CSV file containing XAUUSD 15-min candle data.
    file_path = "XAUUSD_5M_04.04.2022-24.01.2025.csv"
    data = load_data(file_path)
    X, data = prepare_features(data)
    
    # Fit a Gaussian HMM with 2 hidden states.
    model, hidden_states = fit_hmm(X, n_states=2)
    
    # Analyze the hidden states to determine market bias.
    bias, state_stats, state_counts = analyze_states(data, hidden_states)
    print("HMM Analysis Results:")
    print("----------------------")
    print(f"Overall Market Bias: {bias}")
    print("State Statistics (State: Mean Return):")
    for state, mean_return in state_stats.items():
        print(f"  State {state}: {mean_return:.6f}")
    print("State Occurrence Counts:")
    print(state_counts)
    
    # Provide position sizing and risk management recommendations.
    print("\n")
    risk_management_recommendations()

if __name__ == "__main__":
    main()
