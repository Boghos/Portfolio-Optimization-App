import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from scipy import optimize
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                          confusion_matrix, classification_report, roc_curve, 
                          auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sidebar
st.sidebar.title("Portfolio Optimizer")

st.sidebar.header("📊 Investment Tools")


# Define functions from the notebook
@st.cache_data(ttl=24*3600)  # Cache for 24 hours
def get_stock_data(tickers, start_date, end_date):
    """
    Download stock data for the given tickers and date range
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# @st.cache_data
# def get_stock_data(tickers, start_date, end_date):
#     """
#     Return mock stock data instead of downloading from Yahoo Finance
#     """
#     # Create a dictionary of sample stock data
#     mock_data = {}
    
#     # Base dates for the mock data (5 years of daily data)
#     base_dates = pd.date_range(end=datetime.now(), periods=252*5, freq='B')
    
#     # Use MultiIndex like yfinance does
#     multi_index = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], tickers])
#     mock_df = pd.DataFrame(columns=multi_index, index=base_dates)
    
#     # Generate mock data for each ticker
#     for ticker in tickers:
#         # Starting price between $10 and $500
#         base_price = np.random.uniform(10, 500)
        
#         # Generate price series with random walk
#         volatility = np.random.uniform(0.01, 0.04)  # Daily volatility between 1% and 4%
#         returns = np.random.normal(0.0002, volatility, len(base_dates))  # Slight upward drift
#         price_series = base_price * (1 + returns).cumprod()
        
#         # Add some randomness to other columns
#         mock_df[('Close', ticker)] = price_series
#         mock_df[('Open', ticker)] = price_series * np.random.uniform(0.99, 1.01, len(base_dates))
#         mock_df[('High', ticker)] = price_series * np.random.uniform(1.001, 1.03, len(base_dates))
#         mock_df[('Low', ticker)] = price_series * np.random.uniform(0.97, 0.999, len(base_dates))
#         mock_df[('Adj Close', ticker)] = price_series  # Same as close for simplicity
        
#         # Generate realistic volume
#         mock_df[('Volume', ticker)] = np.random.randint(100000, 10000000, len(base_dates))
    
#     # Filter for the requested date range
#     filtered_df = mock_df.loc[(mock_df.index >= pd.Timestamp(start_date)) & 
#                              (mock_df.index <= pd.Timestamp(end_date))]
    
#     return filtered_df

def calculate_portfolio_metrics(stock_data, window=252):
    """
    Calculate portfolio metrics based on Modern Portfolio Theory for each stock
    """
    # Use Close price
    prices = stock_data['Close']
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Market proxy (S&P 500)
    market_returns = returns['^GSPC']
    
    # Portfolio metrics dataframe
    metrics = pd.DataFrame()
    
    # List of stocks (excluding market index)
    stocks = [col for col in returns.columns if col != '^GSPC']
    
    for stock in stocks:
        stock_returns = returns[stock]
        
        # Rolling calculations for different time periods
        for period in [30, 90, 180]:
            if len(stock_returns) < period:
                continue
                
            # Expected return (annualized)
            expected_return = stock_returns.rolling(period).mean() * 252
            
            # Volatility (annualized)
            volatility = stock_returns.rolling(period).std() * np.sqrt(252)
            
            # Calculate rolling beta
            cov_market = stock_returns.rolling(period).cov(market_returns)
            var_market = market_returns.rolling(period).var()
            beta = cov_market / var_market
            
            # Calculate rolling alpha (CAPM)
            risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
            expected_market_return = market_returns.rolling(period).mean() * 252
            theoretical_return = risk_free_rate * 252 + beta * (expected_market_return - risk_free_rate * 252)
            alpha = expected_return - theoretical_return
            
            # Sharpe ratio
            sharpe = (expected_return - risk_free_rate * 252) / volatility
            
            # Calculate maximum drawdown
            cum_returns = (1 + stock_returns).cumprod()
            rolling_max = cum_returns.rolling(period, min_periods=1).max()
            drawdown = cum_returns / rolling_max - 1
            max_drawdown = drawdown.rolling(period, min_periods=1).min()
            
            # Value at Risk (95% confidence)
            var_95 = stock_returns.rolling(period).quantile(0.05)
            
            # Volatility of volatility (measure of uncertainty)
            if period >= 90:
                vol_of_vol = volatility.rolling(period // 3).std()
            else:
                vol_of_vol = pd.Series(np.nan, index=volatility.index)
            
            # Store calculations with timestamp
            for idx, date in enumerate(stock_returns.index[period-1:]):
                i = idx + period - 1
                
                # Skip if we don't have enough future data
                if i + 30 >= len(stock_returns):
                    continue
                
                # Calculate future performance (next 30 days return)
                future_return = (prices.loc[stock_returns.index[i+30], stock] / 
                                 prices.loc[date, stock]) - 1
                
                # Future market return
                future_market_return = (prices.loc[stock_returns.index[i+30], '^GSPC'] / 
                                        prices.loc[date, '^GSPC']) - 1
                
                # Target: whether stock outperforms market in next 30 days
                outperforms = 1 if future_return > future_market_return else 0
                
                # Add row to metrics dataframe
                new_row = {
                    'date': date,
                    'stock': stock,
                    'period': period,
                    'expected_return': expected_return.iloc[i],
                    'volatility': volatility.iloc[i],
                    'beta': beta.iloc[i],
                    'alpha': alpha.iloc[i],
                    'sharpe_ratio': sharpe.iloc[i],
                    'max_drawdown': max_drawdown.iloc[i],
                    'var_95': var_95.iloc[i],
                    'vol_of_vol': vol_of_vol.iloc[i],
                    'market_return_current': market_returns.iloc[i],
                    'future_return': future_return,
                    'future_market_return': future_market_return,
                    'outperforms_market': outperforms
                }
                
                # Use concat instead of append
                metrics = pd.concat([metrics, pd.DataFrame([new_row])], ignore_index=True)
    
    return metrics

def prepare_features(metrics_df):
    """
    Prepare features for machine learning
    """
    # Drop rows with NaN values
    df = metrics_df.dropna()
    
    # Create additional features
    df['return_to_risk'] = df['expected_return'] / df['volatility'].replace(0, 0.0001)  # Avoid division by zero
    df['alpha_to_beta'] = df['alpha'] / df['beta'].apply(lambda x: max(0.01, abs(x)))  # Avoid division by zero
    
    # Convert date to numerical features
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['stock', 'period'], drop_first=True)
    
    # Drop non-feature columns
    X = df.drop(['date', 'future_return', 'future_market_return', 'outperforms_market'], axis=1)
    y = df['outperforms_market']
    
    return X, y

def optimize_portfolio(expected_returns, cov_matrix, risk_tolerance=None, target_return=None):
    """
    Optimize portfolio weights using Modern Portfolio Theory.
    Either specify risk_tolerance (for minimum volatility at that risk level)
    or target_return (for minimum risk at that return level)
    """
    num_assets = len(expected_returns)
    
    # Define constraints (weights sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds for weights (e.g., no short selling)
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess (equal weight)
    initial_guess = np.array([1/num_assets] * num_assets)
    
    if risk_tolerance is not None:
        # Maximize return for a given risk tolerance
        objective = lambda weights: -np.dot(weights, expected_returns)
        risk_constraint = {'type': 'ineq', 
                           'fun': lambda weights: risk_tolerance - np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))}
        constraints = [constraints, risk_constraint]
    else:
        # Minimize risk for a given target return
        objective = lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return_constraint = {'type': 'eq', 
                            'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
        constraints = [constraints, return_constraint]
    
    # Optimize
    result = optimize.minimize(objective, initial_guess, 
                              method='SLSQP', bounds=bounds, 
                              constraints=constraints)
    
    return result['x']

@st.cache_resource
def train_models(X_train_scaled, y_train):
    """Train ML models and return them"""
    # Define models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', random_state=42, probability=True, max_iter=10000),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
    return models

def create_optimized_portfolio(user_stocks, historical_data, risk_preference='moderate', use_ml=True):
    """
    Create an optimized portfolio from user's selected stocks with optional ML enhancement
    
    Parameters:
    - user_stocks: List of stock tickers the user wants to include
    - historical_data: Historical price data for calculations
    - risk_preference: 'conservative', 'moderate', or 'aggressive'
    - use_ml: Whether to use machine learning to enhance the optimization
    
    Returns:
    - Dictionary with optimal weights and expected performance
    """
    try:
        # Use Close price data - handle MultiIndex columns properly
        if isinstance(historical_data.columns, pd.MultiIndex):
            prices = historical_data['Close']
        else:
            # If historical_data is not a MultiIndex DataFrame, extract Close differently
            prices = pd.DataFrame({ticker: historical_data[ticker] for ticker in historical_data.columns})
        
        # Verify stocks exist in the data
        missing_stocks = [stock for stock in user_stocks if stock not in prices.columns]
        if missing_stocks:
            raise ValueError(f"Missing stock data for: {', '.join(missing_stocks)}")
        
        # Filter for user's selected stocks
        stocks_prices = prices[user_stocks]
        
        # Calculate daily returns
        returns = stocks_prices.pct_change().dropna()
        
        # Calculate expected returns
        expected_returns = returns.mean() * 252  # annualized
        
        # If returns look unrealistic, use reasonable defaults
        if (expected_returns < -0.5).any() or (expected_returns > 1.0).any():
            st.warning("Unrealistic returns detected. Using historical market averages.")
            # Use historical average market returns (7% for stocks)
            expected_returns = pd.Series(0.07, index=user_stocks)
        
        # Default probabilities (without ML)
        probabilities = {stock: 0.5 for stock in user_stocks}
        
        # If ML is enabled
        if use_ml:
            st.info("Using machine learning to enhance portfolio optimization...")
            
            try:
                # A simplified ML approach that doesn't rely on the complicated feature engineering
                ml_probabilities = {}
                
                for stock in user_stocks:
                    # Get stock data for analysis
                    stock_prices = stocks_prices[stock]
                    stock_returns = stock_prices.pct_change().dropna()
                    
                    # Calculate some basic metrics for ML features
                    
                    # 1. Recent performance (last 30 days vs previous 30 days)
                    if len(stock_returns) >= 60:
                        recent_return = stock_returns[-30:].mean()
                        previous_return = stock_returns[-60:-30].mean()
                        momentum = recent_return > previous_return
                    else:
                        momentum = True  # Default
                    
                    # 2. Volatility trend (is volatility decreasing?)
                    if len(stock_returns) >= 60:
                        recent_vol = stock_returns[-30:].std()
                        previous_vol = stock_returns[-60:-30].std()
                        vol_improving = recent_vol < previous_vol
                    else:
                        vol_improving = True  # Default
                    
                    # 3. Recent performance vs market
                    if '^GSPC' in prices.columns:
                        market_returns = prices['^GSPC'].pct_change().dropna()
                        if len(stock_returns) >= 30 and len(market_returns) >= 30:
                            stock_monthly = stock_returns[-30:].mean()
                            market_monthly = market_returns[-30:].mean()
                            outperforming = stock_monthly > market_monthly
                        else:
                            outperforming = True  # Default
                    else:
                        outperforming = True  # Default
                    
                    # 4. Sharpe ratio (return to risk ratio)
                    annual_return = stock_returns.mean() * 252
                    annual_vol = stock_returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    good_sharpe = sharpe > 0.5  # Arbitrary threshold
                    
                    # Calculate outperformance probability based on these factors
                    # Simple scoring system (each factor contributes 0.1 to the base probability of 0.5)
                    prob = 0.5
                    if momentum:
                        prob += 0.1
                    if vol_improving:
                        prob += 0.1
                    if outperforming:
                        prob += 0.1
                    if good_sharpe:
                        prob += 0.1
                        
                    # Cap between 0.3 and 0.7 to avoid extreme adjustments
                    prob = max(0.3, min(0.7, prob))
                    
                    ml_probabilities[stock] = prob
                
                # Update probabilities
                probabilities = ml_probabilities
                
                # Display ML predictions
                st.write("ML-enhanced outperformance estimates:")
                prob_df = pd.DataFrame({
                    'Stock': list(probabilities.keys()),
                    'Outperformance Probability': [f"{prob*100:.2f}%" for prob in probabilities.values()]
                })
                # Set the index to start from 1 instead of 0
                prob_df.index = range(1, len(prob_df) + 1)
                st.dataframe(prob_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"ML prediction error: {str(e)}")
                st.warning("Falling back to traditional portfolio optimization.")
        
        # Adjust expected returns by prediction confidence
        confidence_adjusted_returns = {}
        for stock in user_stocks:
            # Apply a scaling to avoid extreme adjustments
            adjustment_factor = 0.5 + (probabilities[stock] - 0.5) * 0.8
            confidence_adjusted_returns[stock] = expected_returns[stock] * adjustment_factor
        
        # Convert to numpy array for optimization
        expected_returns_array = np.array([confidence_adjusted_returns[stock] for stock in user_stocks])
        
        # Calculate covariance matrix from a longer history (e.g., 1 year)
        cov_matrix = returns.iloc[-252:].cov() * 252  # Annualized
        
        # Define risk preference mappings
        risk_mappings = {
            'conservative': {'target_return': None, 'risk_tolerance': 0.1},  # Min vol
            'moderate': {'target_return': None, 'risk_tolerance': None},     # Max Sharpe
            'aggressive': {'target_return': 0.15, 'risk_tolerance': None}    # Target 15% return
        }
        
        params = risk_mappings.get(risk_preference, risk_mappings['moderate'])
        
        # For max Sharpe ratio (moderate risk)
        if risk_preference == 'moderate':
            results = []
            for target_return in np.linspace(0.05, 0.25, 50):
                try:
                    weights = optimize_portfolio(expected_returns_array, cov_matrix.values, 
                                             target_return=target_return)
                    portfolio_return = np.sum(expected_returns_array * weights)
                    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                    results.append({'weights': weights, 'return': portfolio_return, 
                                  'volatility': portfolio_vol, 'sharpe': sharpe})
                except:
                    continue
            
            if not results:
                st.warning("Failed to find optimal portfolio with max Sharpe. Using equal weights.")
                weights = np.array([1/len(user_stocks)] * len(user_stocks))
            else:
                # Find weights with max Sharpe
                optimal = max(results, key=lambda x: x['sharpe'])
                weights = optimal['weights']
        else:
            # Conservative or aggressive portfolios
            try:
                weights = optimize_portfolio(
                    expected_returns_array, 
                    cov_matrix.values,
                    risk_tolerance=params['risk_tolerance'],
                    target_return=params['target_return']
                )
            except Exception as e:
                st.error(f"Optimization error: {e}")
                st.warning("Using equal weights.")
                weights = np.array([1/len(user_stocks)] * len(user_stocks))
        
        # Create result dictionary with stock weights
        portfolio = {stock: weight for stock, weight in zip(user_stocks, weights)}
        
        # Calculate expected portfolio metrics
        expected_return = sum(confidence_adjusted_returns[stock] * weight for stock, weight in portfolio.items())
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        expected_sharpe = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Create portfolio metrics
        metrics = {
            'portfolio_weights': portfolio,
            'expected_annual_return': expected_return,
            'expected_volatility': portfolio_volatility,
            'expected_sharpe_ratio': expected_sharpe,
            'stock_returns': {stock: expected_returns[stock] for stock in user_stocks},
            'ml_probabilities': probabilities if use_ml else None
        }
        
        return metrics
    
    except Exception as e:
        st.error(f"Portfolio optimization error: {str(e)}")
        raise
       

def plot_portfolio_allocation(portfolio_weights):
    """Create a pie chart of portfolio weights"""
    # Sort by weight (descending)
    sorted_weights = {k: v for k, v in sorted(portfolio_weights.items(), key=lambda item: item[1], reverse=True)}
    
    # Filter out stocks with very small weights (less than 1%)
    filtered_weights = {k: v for k, v in sorted_weights.items() if v >= 0.01}
    other_weight = sum(v for k, v in sorted_weights.items() if v < 0.01)
    
    if other_weight > 0:
        filtered_weights['Other'] = other_weight
    
    # Make sure to close any existing figures first
    plt.close('all')
    
    # Create a completely new figure with a specific figure number
    fig = plt.figure(num=1, figsize=(10, 8), clear=True)
    ax = fig.add_subplot(111)
    
    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        list(filtered_weights.values()), 
        labels=list(filtered_weights.keys()),
        autopct='%1.1f%%', 
        startangle=90, 
        shadow=True,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        textprops={'fontsize': 12}
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add legend
    ax.legend(wedges, filtered_weights.keys(), 
              title="Stocks",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig

def plot_efficient_frontier(returns, stock_data, selected_stocks, optimal_portfolio=None):
    """Plot the efficient frontier with the optimal portfolio"""
    # Get close prices for selected stocks
    prices = stock_data['Close'][selected_stocks]
    
    # Calculate returns and covariance
    stock_returns = prices.pct_change().dropna()
    mean_returns = stock_returns.mean() * 252
    cov_matrix = stock_returns.cov() * 252
    
    # Generate random portfolios
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(selected_stocks))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = portfolio_return / portfolio_std_dev
        
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio
    
    # Create figure - close all existing figures first
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot random portfolios
    scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], 
                         cmap='viridis', marker='o', s=10, alpha=0.3)
    
    # Plot individual stocks
    for i, stock in enumerate(selected_stocks):
        stock_return = mean_returns[stock]
        stock_risk = np.sqrt(cov_matrix.loc[stock, stock])
        ax.scatter(stock_risk, stock_return, marker='o', s=100, 
                  label=stock, edgecolors='black')
        ax.annotate(stock, (stock_risk, stock_return), xytext=(5, 5), 
                   textcoords='offset points', fontsize=10)
    
    # Plot optimal portfolio if provided
    if optimal_portfolio:
        opt_return = optimal_portfolio['expected_annual_return']
        opt_vol = optimal_portfolio['expected_volatility']
        ax.scatter(opt_vol, opt_return, s=300, color='red', marker='*', 
                  label='Optimal Portfolio', edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio')
    
    # Labels and title
    ax.set_xlabel('Annualized Volatility (Standard Deviation)')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Efficient Frontier of Portfolios')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    
    plt.tight_layout()
    return fig

def plot_stock_comparison(stock_data, selected_stocks, period=365):
    """Plot comparative stock performance for the last year"""
    # Get close prices
    prices = stock_data['Close'][selected_stocks].tail(period)
    
    # Calculate normalized returns (starting from 100)
    normalized = 100 * prices / prices.iloc[0]
    
    # Plot - close all existing figures first
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    for column in normalized.columns:
        ax.plot(normalized.index, normalized[column], linewidth=2, label=column)
    
    ax.set_title('Stock Price Performance Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (Base 100)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_return_vs_risk(stock_returns, selected_stocks):
    """Create a scatter plot of return vs risk for selected stocks"""
    # Extract annualized returns and volatility
    returns_data = []
    
    for stock in selected_stocks:
        if stock in stock_returns:  # Check if stock exists in returns
            returns_data.append({
                'Stock': stock, 
                'Return': stock_returns[stock] * 100, 
                'Volatility': np.sqrt(stock_returns[stock]) * 100  # Simplified
            })
    
    df = pd.DataFrame(returns_data)
    
    if df.empty:
        return None
    
    # Create plot - close all existing figures first
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    scatter = ax.scatter(df['Volatility'], df['Return'], s=100, alpha=0.7)
    
    # Add labels to each point
    for i, row in df.iterrows():
        ax.annotate(row['Stock'], 
                   (row['Volatility'], row['Return']),
                   xytext=(5, 5),
                   textcoords="offset points",
                   fontsize=10)
    
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Expected Annual Return (%)')
    ax.set_title('Return vs. Risk for Selected Stocks')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
# Add these additional backtest functions to enhance the capabilities

def backtest_portfolio(stock_data, weights, start_date=None, end_date=None, initial_investment=10000, rebalance_frequency="monthly", benchmark_type="S&P 500"):
    """
    Backtest a portfolio with fixed weights over a historical period
    
    Parameters:
    - stock_data: DataFrame with historical price data (must have 'Close' column)
    - weights: Dictionary mapping stock tickers to their allocation weights (must sum to 1)
    - start_date: Start date for backtest (defaults to first date in data)
    - end_date: End date for backtest (defaults to last date in data)
    - initial_investment: Initial portfolio value (default $10,000)
    - rebalance_frequency: How often to rebalance ('monthly', 'quarterly', 'yearly', 'never')
    - benchmark_type: Type of benchmark to use ('S&P 500', 'Equal Weight', or None)
    
    Returns:
    - Dictionary with backtest results (portfolio values, returns, metrics)
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Extract close prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data['Close']
    else:
        prices = stock_data
    
    # Filter for tickers in our weights
    tickers = list(weights.keys())
    prices = prices[tickers]
    
    # Set date range if specified
    if start_date:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
    if end_date:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date)]
    
    # Make sure we have data
    if len(prices) == 0:
        raise ValueError("No price data found for the specified date range")
    
    # Setup for backtest
    portfolio_values = pd.DataFrame(index=prices.index)
    portfolio_values['Portfolio Value'] = 0
    
    # Calculate daily returns
    returns = prices.pct_change().fillna(0)
    
    # Create a dict to track shares for each stock
    shares = {ticker: 0 for ticker in tickers}
    
    # Helper function to calculate holdings
    def calculate_holdings(current_date, rebalance=False):
        nonlocal shares, initial_investment
        
        # Get current prices
        current_prices = prices.loc[current_date]
        
        if rebalance or all(shares[ticker] == 0 for ticker in tickers):
            # Rebalance or initial investment
            current_portfolio_value = sum(shares[ticker] * current_prices[ticker] for ticker in tickers)
            if current_portfolio_value == 0:
                current_portfolio_value = initial_investment
                
            # Calculate target value for each stock
            for ticker in tickers:
                target_value = current_portfolio_value * weights[ticker]
                if current_prices[ticker] > 0:  # Avoid division by zero
                    shares[ticker] = target_value / current_prices[ticker]
        
        # Calculate current portfolio value
        current_value = sum(shares[ticker] * current_prices[ticker] for ticker in tickers)
        return current_value
    
    # Initialize tracking variables for rebalancing
    last_rebalance_date = None
    
    # Set up rebalancing intervals
    if rebalance_frequency == "monthly":
        months_delta = 1
    elif rebalance_frequency == "quarterly":
        months_delta = 3
    elif rebalance_frequency == "yearly":
        months_delta = 12
    
    # Run backtest day by day
    for date in prices.index:
        if pd.isna(prices.loc[date]).any():
            # Skip dates with missing data
            continue
            
        should_rebalance = False
        
        # Check if we need to rebalance
        if rebalance_frequency != "never":
            if last_rebalance_date is None:
                should_rebalance = True
            elif rebalance_frequency == "monthly":
                if (date.year > last_rebalance_date.year) or (date.month > last_rebalance_date.month):
                    should_rebalance = True
            elif rebalance_frequency == "quarterly":
                if (date.year > last_rebalance_date.year) or (date.month - last_rebalance_date.month >= 3):
                    should_rebalance = True
            elif rebalance_frequency == "yearly":
                if date.year > last_rebalance_date.year:
                    should_rebalance = True
                    
        # Calculate portfolio value
        portfolio_value = calculate_holdings(date, rebalance=should_rebalance)
        portfolio_values.loc[date, 'Portfolio Value'] = portfolio_value
        
        # Update last rebalance date if we rebalanced
        if should_rebalance:
            last_rebalance_date = date
    
    # Calculate portfolio statistics
    portfolio_returns = portfolio_values['Portfolio Value'].pct_change().fillna(0)
    
    # Initialize market benchmark
    market_benchmark = None
    
    # Add benchmark comparison based on selected type
    if benchmark_type == "S&P 500":
        # Check if S&P 500 data is available
        has_sp500 = False
        try:
            if isinstance(stock_data.columns, pd.MultiIndex):
                has_sp500 = '^GSPC' in stock_data['Close'].columns
            else:
                has_sp500 = '^GSPC' in stock_data.columns
                
            if has_sp500:
                # Get S&P 500 prices
                if isinstance(stock_data.columns, pd.MultiIndex):
                    market_prices = stock_data['Close']['^GSPC']
                else:
                    market_prices = stock_data['^GSPC']
                
                # Filter for date range
                market_prices = market_prices.loc[
                    (market_prices.index >= min(portfolio_values.index)) & 
                    (market_prices.index <= max(portfolio_values.index))
                ]
                
                if len(market_prices) > 0:
                    # Calculate market returns and cumulative performance
                    market_returns = market_prices.pct_change().fillna(0)
                    market_cumulative = (1 + market_returns).cumprod()
                    
                    # Create benchmark DataFrame
                    market_benchmark = pd.DataFrame(index=portfolio_values.index)
                    
                    # Align the index with portfolio values
                    common_dates = market_cumulative.index.intersection(portfolio_values.index)
                    if len(common_dates) > 0:
                        # Normalize to the same starting value as portfolio
                        start_value = portfolio_values['Portfolio Value'].iloc[0]
                        market_start = market_cumulative.loc[common_dates[0]]
                        
                        # Calculate benchmark values for all common dates
                        for date in common_dates:
                            if date in market_cumulative.index:
                                relative_perf = market_cumulative.loc[date] / market_start
                                market_benchmark.loc[date, 'S&P 500'] = start_value * relative_perf
        except Exception as e:
            print(f"Error setting up S&P 500 benchmark: {e}")
            
    elif benchmark_type == "Equal Weight" and benchmark_type is not None:
        try:
            # Create equal weight allocation for the same stocks
            equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
            
            # Run a separate backtest with equal weights
            # Use a recursive call but with benchmark_type=None to avoid infinite recursion
            equal_weight_results = backtest_portfolio(
                stock_data,
                equal_weights,
                start_date=start_date,
                end_date=end_date,
                initial_investment=initial_investment,
                rebalance_frequency=rebalance_frequency,
                benchmark_type=None  # Prevent infinite recursion
            )
            
            # Extract the equal weight portfolio values
            if 'portfolio_values' in equal_weight_results:
                market_benchmark = pd.DataFrame(index=portfolio_values.index)
                market_benchmark['Equal Weight'] = equal_weight_results['portfolio_values']['Portfolio Value']
        except Exception as e:
            print(f"Error setting up Equal Weight benchmark: {e}")
    
    # Calculate metrics
    total_days = len(portfolio_returns)
    annual_factor = 252 / max(total_days, 1)  # Annualization factor, avoid division by zero
    
    total_return = (portfolio_values['Portfolio Value'].iloc[-1] / portfolio_values['Portfolio Value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** annual_factor - 1 if total_days > 0 else 0
    
    volatility = portfolio_returns.std() * np.sqrt(252) if total_days > 1 else 0  # Annualized
    
    # Calculate Sharpe Ratio (assuming 0% risk-free rate for simplicity)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Calculate drawdowns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / running_max) - 1
    max_drawdown = drawdowns.min()
    
    # Alpha and Beta (if market data available)
    alpha, beta = None, None
    if market_benchmark is not None:
        # Get benchmark column name
        benchmark_col = market_benchmark.columns[0] if len(market_benchmark.columns) > 0 else None
        
        if benchmark_col is not None:
            # Calculate benchmark returns
            benchmark_values = market_benchmark[benchmark_col].dropna()
            benchmark_returns = benchmark_values.pct_change().fillna(0)
            
            # Get matching dates
            matching_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(matching_dates) > 10:  # Need enough data points
                portfolio_subset = portfolio_returns.loc[matching_dates]
                market_subset = benchmark_returns.loc[matching_dates]
                
                # Calculate beta
                covariance = portfolio_subset.cov(market_subset)
                market_variance = market_subset.var()
                beta = covariance / market_variance if market_variance > 0 else 1
                
                # Calculate alpha (annualized)
                market_return_annual = (1 + market_subset.mean()) ** 252 - 1
                portfolio_return_annual = (1 + portfolio_subset.mean()) ** 252 - 1
                alpha = portfolio_return_annual - (0.0 + beta * (market_return_annual - 0.0))  # Using 0% risk-free rate
    
    # Monthly returns analysis
    monthly_returns = pd.DataFrame()
    if len(portfolio_returns) > 30:
        monthly_returns = portfolio_returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        positive_months = (monthly_returns > 0).sum() / len(monthly_returns)
    else:
        best_month = worst_month = positive_months = None
    
    # Results dictionary with all metrics and time series
    results = {
        'portfolio_values': portfolio_values,
        'market_benchmark': market_benchmark,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'alpha': alpha,
        'beta': beta,
        'best_month': best_month,
        'worst_month': worst_month,
        'positive_months_ratio': positive_months,
        'daily_returns': portfolio_returns,
        'monthly_returns': monthly_returns
    }
    
    return results

def create_backtest_metrics_table(backtest_results):
    """
    Create a formatted DataFrame with backtest performance metrics
    
    Parameters:
    - backtest_results: Dictionary with backtest data from backtest_portfolio function
    
    Returns:
    - DataFrame with formatted metrics
    """
    import pandas as pd
    
    # Create a dictionary of metrics
    metrics = {
        'Total Return': f"{backtest_results['total_return']*100:.2f}%",
        'Annualized Return': f"{backtest_results['annualized_return']*100:.2f}%",
        'Volatility (Annualized)': f"{backtest_results['volatility']*100:.2f}%",
        'Sharpe Ratio': f"{backtest_results['sharpe_ratio']:.2f}",
        'Maximum Drawdown': f"{backtest_results['max_drawdown']*100:.2f}%"
    }
    
    # Add alpha and beta if available
    if backtest_results['alpha'] is not None:
        metrics['Alpha'] = f"{backtest_results['alpha']*100:.2f}%"
    if backtest_results['beta'] is not None:
        metrics['Beta'] = f"{backtest_results['beta']:.2f}"
    
    # Add monthly statistics if available
    if backtest_results['best_month'] is not None:
        metrics['Best Month'] = f"{backtest_results['best_month']*100:.2f}%"
        metrics['Worst Month'] = f"{backtest_results['worst_month']*100:.2f}%"
        metrics['Positive Months'] = f"{backtest_results['positive_months_ratio']*100:.1f}%"
    
    # Create DataFrame and format
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })

    # Set the index to start from 1 instead of 0
    metrics_df.index = range(1, len(metrics_df) + 1)
    
    return metrics_df

def plot_backtest_performance(backtest_results):
    """
    Create plots to visualize backtest performance
    
    Parameters:
    - backtest_results: Dictionary with backtest data from backtest_portfolio function
    
    Returns:
    - Dictionary with matplotlib figures for different plots
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    plots = {}
    
    # Close all existing figures first
    plt.close('all')
    
    # 1. Portfolio Value Plot
    fig_value = plt.figure(figsize=(12, 6))
    ax_value = fig_value.add_subplot(111)
    
    # Ensure data is numeric and handle any missing values
    portfolio_values = backtest_results['portfolio_values']['Portfolio Value'].astype(float)
    
    # Plot portfolio value
    ax_value.plot(portfolio_values.index, portfolio_values.values,
                 label='Portfolio', linewidth=2)
    
    # Plot benchmark if available
    if backtest_results['market_benchmark'] is not None:
        benchmark_df = backtest_results['market_benchmark']
        if not benchmark_df.empty and len(benchmark_df.columns) > 0:
            benchmark_col = benchmark_df.columns[0]
            benchmark_values = benchmark_df[benchmark_col].dropna().astype(float)
            
            if len(benchmark_values) > 0:
                ax_value.plot(benchmark_values.index, benchmark_values.values,
                             label=benchmark_col, linewidth=1.5, alpha=0.7, linestyle='--')
    
    ax_value.set_title('Portfolio Performance Backtest')
    ax_value.set_xlabel('Date')
    ax_value.set_ylabel('Value ($)')
    ax_value.grid(True, alpha=0.3)
    ax_value.legend()
    plots['value_plot'] = fig_value
    
    # 2. Drawdown Plot
    portfolio_returns = backtest_results['daily_returns'].astype(float)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns / running_max) - 1
    
    fig_drawdown = plt.figure(figsize=(12, 5))
    ax_drawdown = fig_drawdown.add_subplot(111)
    ax_drawdown.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.3)
    ax_drawdown.plot(drawdowns.index, drawdowns.values, color='red', alpha=0.8, label='Portfolio Drawdown')
    
    # Add benchmark drawdown if available
    if backtest_results['market_benchmark'] is not None:
        benchmark_df = backtest_results['market_benchmark']
        if not benchmark_df.empty and len(benchmark_df.columns) > 0:
            benchmark_col = benchmark_df.columns[0]
            benchmark_values = benchmark_df[benchmark_col].dropna().astype(float)
            
            if len(benchmark_values) > 0:
                # Calculate benchmark drawdowns
                bench_returns = benchmark_values.pct_change().fillna(0)
                bench_cumulative = (1 + bench_returns).cumprod()
                bench_max = bench_cumulative.cummax()
                bench_drawdowns = (bench_cumulative / bench_max) - 1
                
                ax_drawdown.plot(bench_drawdowns.index, bench_drawdowns.values, 
                                color='blue', alpha=0.6, linestyle='--',
                                label=f'{benchmark_col} Drawdown')
                ax_drawdown.legend()
    
    ax_drawdown.set_title('Portfolio Drawdown')
    ax_drawdown.set_xlabel('Date')
    ax_drawdown.set_ylabel('Drawdown (%)')
    ax_drawdown.grid(True, alpha=0.3)
    plots['drawdown_plot'] = fig_drawdown
    
    # 3. Relative Performance Plot (if benchmark available)
    if backtest_results['market_benchmark'] is not None:
        benchmark_df = backtest_results['market_benchmark']
        if not benchmark_df.empty and len(benchmark_df.columns) > 0:
            benchmark_col = benchmark_df.columns[0]
            benchmark_values = benchmark_df[benchmark_col].dropna().astype(float)
            
            if len(benchmark_values) > 0:
                # Get common dates
                common_dates = portfolio_values.index.intersection(benchmark_values.index)
                
                if len(common_dates) > 0:
                    # Create relative performance chart
                    fig_relative = plt.figure(figsize=(12, 5))
                    ax_relative = fig_relative.add_subplot(111)
                    
                    # Calculate relative performance
                    portfolio_subset = portfolio_values.loc[common_dates]
                    benchmark_subset = benchmark_values.loc[common_dates]
                    
                    # Normalize to 100 at start
                    norm_portfolio = 100 * portfolio_subset / portfolio_subset.iloc[0]
                    norm_benchmark = 100 * benchmark_subset / benchmark_subset.iloc[0]
                    
                    # Plot normalized values
                    ax_relative.plot(common_dates, norm_portfolio, 
                                    label='Portfolio', linewidth=2)
                    ax_relative.plot(common_dates, norm_benchmark, 
                                    label=benchmark_col, linewidth=1.5, 
                                    linestyle='--', alpha=0.7)
                    
                    # Calculate and plot relative performance
                    relative_perf = norm_portfolio / norm_benchmark * 100 - 100
                    
                    ax2 = ax_relative.twinx()
                    ax2.plot(common_dates, relative_perf, 
                            label='Relative Performance', 
                            color='green', alpha=0.5)
                    ax2.set_ylabel(f'Outperformance vs {benchmark_col} (%)')
                    
                    # Add horizontal line at 0%
                    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    
                    ax_relative.set_title(f'Portfolio vs {benchmark_col}')
                    ax_relative.set_xlabel('Date')
                    ax_relative.set_ylabel('Normalized Value (Base 100)')
                    ax_relative.grid(True, alpha=0.3)
                    
                    # Combine legends from both axes
                    lines1, labels1 = ax_relative.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax_relative.legend(lines1 + lines2, labels1 + labels2, loc='best')
                    
                    plots['relative_performance'] = fig_relative
    
    # 4. Monthly Returns Heatmap
    if backtest_results['monthly_returns'] is not None and len(backtest_results['monthly_returns']) > 3:
        try:
            # Reshape data for heatmap
            monthly_data = backtest_results['monthly_returns'].copy().astype(float)
            monthly_data.index = pd.to_datetime(monthly_data.index)
            
            # Create a new DataFrame with years as rows and months as columns
            years = sorted(set(monthly_data.index.year))
            months = range(1, 13)
            
            heatmap_data = pd.DataFrame(index=years, columns=months)
            
            # Fill with monthly returns
            for idx, value in monthly_data.items():
                year = idx.year
                month = idx.month
                if year in years and month in months:
                    heatmap_data.loc[year, month] = value
            
            # Create heatmap
            fig_heatmap = plt.figure(figsize=(12, len(years)*0.8))
            ax_heatmap = fig_heatmap.add_subplot(111)
            
            # Use a diverging colormap: red for negative, green for positive
            cmap = plt.cm.RdYlGn  # Red-Yellow-Green
            
            # Convert to numpy array and ensure it's all float type
            heatmap_array = heatmap_data.values.astype(float)
            
            # Find the maximum absolute return for symmetrical color scaling
            with np.errstate(invalid='ignore'):  # Ignore NaN warnings
                max_abs_return = np.nanmax(np.abs(heatmap_array))
            if np.isnan(max_abs_return) or max_abs_return == 0:
                max_abs_return = 0.05  # Default if all NaN or zeros
            
            # Create heatmap with a colorbar
            im = ax_heatmap.imshow(heatmap_array, cmap=cmap, aspect='auto', 
                                 vmin=-max_abs_return, vmax=max_abs_return)
            
            # Format
            ax_heatmap.set_title('Monthly Returns Heatmap')
            ax_heatmap.set_xlabel('Month')
            ax_heatmap.set_ylabel('Year')
            
            # Set x and y ticks
            ax_heatmap.set_xticks(range(len(months)))
            ax_heatmap.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax_heatmap.set_yticks(range(len(years)))
            ax_heatmap.set_yticklabels(years)
            
            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax_heatmap)
            cbar.set_label('Monthly Return (%)')
            
            # Add text annotations with return values
            for i in range(len(years)):
                for j in range(len(months)):
                    value = heatmap_data.iloc[i, j]
                    if not pd.isna(value):
                        text_color = 'black' if abs(value) < max_abs_return/2 else 'white'
                        text = f"{value*100:.1f}%" if not pd.isna(value) else ""
                        ax_heatmap.text(j, i, text, ha='center', va='center', color=text_color)
            
            plt.tight_layout()
            plots['monthly_heatmap'] = fig_heatmap
        except Exception as e:
            print(f"Error creating monthly heatmap: {e}")
    
    return plots



def compare_strategies(stock_data, strategies, start_date=None, end_date=None, initial_investment=10000):
    """
    Compare multiple portfolio strategies
    
    Parameters:
    - stock_data: DataFrame with historical price data
    - strategies: Dictionary mapping strategy names to weight dictionaries
    - start_date, end_date: Date range for backtest
    - initial_investment: Starting amount for each strategy
    
    Returns:
    - Dictionary with comparative results
    """
    results = {}
    
    # Run backtest for each strategy
    for name, weights in strategies.items():
        results[name] = backtest_portfolio(
            stock_data, weights, start_date, end_date, initial_investment
        )
    
    return results

def plot_strategy_comparison(strategy_results):
    """
    Plot comparative performance of different strategies
    
    Parameters:
    - strategy_results: Dictionary with backtest results from compare_strategies
    
    Returns:
    - Dictionary with matplotlib figures
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    plots = {}
    
    # Get strategy names
    strategy_names = list(strategy_results.keys())
    
    # 1. Value comparison chart
    fig_value, ax_value = plt.subplots(figsize=(12, 6))
    
    # Combine all portfolio values
    performance_df = pd.DataFrame()
    
    for name in strategy_names:
        performance_df[name] = strategy_results[name]['portfolio_values']['Portfolio Value']
    
    # Add benchmark if available
    if 'market_benchmark' in strategy_results[strategy_names[0]] and strategy_results[strategy_names[0]]['market_benchmark'] is not None:
        performance_df['S&P 500'] = strategy_results[strategy_names[0]]['market_benchmark']['S&P 500']
    
    # Plot values
    for column in performance_df.columns:
        ax_value.plot(performance_df.index, performance_df[column], label=column, linewidth=2)
    
    ax_value.set_title('Strategy Performance Comparison')
    ax_value.set_xlabel('Date')
    ax_value.set_ylabel('Value ($)')
    ax_value.grid(True, alpha=0.3)
    ax_value.legend()
    plots['value_comparison'] = fig_value
    
    # 2. Drawdown comparison
    fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
    
    # Calculate drawdowns for each strategy
    for name in strategy_names:
        returns = strategy_results[name]['daily_returns']
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / running_max) - 1
        ax_dd.plot(drawdowns.index, drawdowns, label=name, alpha=0.7)
    
    ax_dd.set_title('Strategy Drawdown Comparison')
    ax_dd.set_xlabel('Date')
    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend()
    plots['drawdown_comparison'] = fig_dd
    
    # 3. Metrics comparison table
    metrics_comparison = pd.DataFrame(index=strategy_names)
    
    # Add metrics for each strategy
    metrics_comparison['Total Return (%)'] = [
        strategy_results[name]['total_return'] * 100 for name in strategy_names
    ]
    metrics_comparison['Annualized Return (%)'] = [
        strategy_results[name]['annualized_return'] * 100 for name in strategy_names
    ]
    metrics_comparison['Volatility (%)'] = [
        strategy_results[name]['volatility'] * 100 for name in strategy_names
    ]
    metrics_comparison['Sharpe Ratio'] = [
        strategy_results[name]['sharpe_ratio'] for name in strategy_names
    ]
    metrics_comparison['Max Drawdown (%)'] = [
        strategy_results[name]['max_drawdown'] * 100 for name in strategy_names
    ]
    
    # Create a bar chart for key metrics
    fig_metrics, ax_metrics = plt.subplots(figsize=(14, 8))
    
    # Reshape data for plotting
    metrics_to_plot = ['Annualized Return (%)', 'Volatility (%)', 'Sharpe Ratio']
    plot_data = metrics_comparison[metrics_to_plot].copy()
    
    # Adjust Sharpe Ratio scale for better visualization
    if 'Sharpe Ratio' in plot_data.columns:
        plot_data['Sharpe Ratio'] = plot_data['Sharpe Ratio'] * 5  # Scale for visibility
    
    # Plot bars
    bar_width = 0.25
    positions = np.arange(len(strategy_names))
    
    for i, metric in enumerate(metrics_to_plot):
        ax_metrics.bar(
            positions + i*bar_width - bar_width, 
            plot_data[metric],
            width=bar_width,
            label=metric
        )
    
    # Formatting
    ax_metrics.set_title('Strategy Performance Metrics Comparison')
    ax_metrics.set_xticks(positions)
    ax_metrics.set_xticklabels(strategy_names)
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, metric in enumerate(metrics_to_plot):
        for j, value in enumerate(plot_data[metric]):
            if metric == 'Sharpe Ratio':
                # Unscale for display
                display_value = value / 5
                ax_metrics.text(
                    j + i*bar_width - bar_width,
                    value + 0.5,
                    f'{display_value:.2f}',
                    ha='center'
                )
            else:
                ax_metrics.text(
                    j + i*bar_width - bar_width,
                    value + 0.5,
                    f'{value:.2f}',
                    ha='center'
                )
    
    plots['metrics_comparison'] = fig_metrics
    plots['metrics_table'] = metrics_comparison
    
    return plots

def generate_risk_strategy(stock_data, selected_stocks, risk_level):
    """
    Generate a portfolio strategy based on the risk level
    
    Parameters:
    - stock_data: DataFrame with historical price data
    - selected_stocks: List of stocks to include
    - risk_level: 'low', 'moderate', or 'high'
    
    Returns:
    - Dictionary with portfolio weights
    """
    import pandas as pd
    import numpy as np
    
    # Extract prices
    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data['Close'][selected_stocks]
    else:
        prices = stock_data[selected_stocks]
    
    # Calculate returns and risk metrics
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean() * 252  # annualized
    cov_matrix = returns.cov() * 252  # annualized
    volatilities = np.sqrt(np.diag(cov_matrix))
    
    # Create a dataframe with risk metrics
    risk_df = pd.DataFrame({
        'Stock': selected_stocks,
        'Expected Return': expected_returns,
        'Volatility': volatilities
    })
    
    # Sort by the appropriate risk metric
    if risk_level == 'low':
        # Low risk - prioritize low volatility stocks
        risk_df = risk_df.sort_values('Volatility')
        weights = {}
        
        # Allocate more to lower volatility stocks
        total_weight = sum(1/vol for vol in risk_df['Volatility'] if vol > 0)
        for _, row in risk_df.iterrows():
            if row['Volatility'] > 0:
                weights[row['Stock']] = (1/row['Volatility']) / total_weight
            else:
                weights[row['Stock']] = 0
    
    elif risk_level == 'high':
        # High risk - prioritize high return stocks
        risk_df = risk_df.sort_values('Expected Return', ascending=False)
        weights = {}
        
        # Allocate more to higher expected return stocks
        total = sum(max(0, ret) for ret in risk_df['Expected Return'])
        if total > 0:
            for _, row in risk_df.iterrows():
                weights[row['Stock']] = max(0, row['Expected Return']) / total
        else:
            # Equal weights if all returns are negative/zero
            weights = {stock: 1/len(selected_stocks) for stock in selected_stocks}
    
    else:  # moderate
        # Moderate risk - balance return and risk using Sharpe ratio
        risk_df['Sharpe'] = risk_df['Expected Return'] / risk_df['Volatility'].replace(0, 0.0001)
        risk_df = risk_df.sort_values('Sharpe', ascending=False)
        weights = {}
        
        # Allocate based on Sharpe ratio
        total = sum(max(0, sharpe) for sharpe in risk_df['Sharpe'])
        if total > 0:
            for _, row in risk_df.iterrows():
                weights[row['Stock']] = max(0, row['Sharpe']) / total
        else:
            # Equal weights if all Sharpe ratios are negative/zero
            weights = {stock: 1/len(selected_stocks) for stock in selected_stocks}
    
    return weights

def perform_stress_test(portfolio_weights, stock_data, scenario):
    """
    Perform a stress test on a portfolio for different market scenarios
    
    Parameters:
    - portfolio_weights: Dictionary of stock weights
    - stock_data: DataFrame with historical price data
    - scenario: String identifying the stress scenario ('financial_crisis', 'covid_crash', 'tech_bubble', 'custom')
    
    Returns:
    - Dictionary with stress test results
    """
    import pandas as pd
    import numpy as np
    
    # Define historical stress periods
    stress_periods = {
        'covid_crash': ('2020-02-19', '2020-03-23'),       # COVID-19 Market Crash
        'inflation_2022': ('2022-01-01', '2022-10-01'),    # 2022 Inflation Surge
        'custom': None  # To be defined
    }
    
    # Get period dates
    if scenario not in stress_periods:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    period = stress_periods[scenario]
    
    # For custom scenario, return a template
    if scenario == 'custom':
        # Create a template for custom scenarios
        tickers = list(portfolio_weights.keys())
        template = pd.DataFrame({
            'Stock': tickers,
            'Price Change (%)': [0] * len(tickers)
        })
        
        return {
            'scenario': 'custom',
            'template': template,
            'description': 'Custom stress test scenario'
        }
    
    # Extract prices for the period
    start_date, end_date = period
    
    # Filter data for the stress period
    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data['Close']
    else:
        prices = stock_data
    
    # Get start and end points (handling weekends/holidays)
    try:
        # Find closest available dates
        dates = prices.index
        start_idx = dates[dates >= start_date][0]
        end_idx = dates[dates <= end_date][-1]
        
        period_prices = prices.loc[start_idx:end_idx]
    except (IndexError, KeyError):
        raise ValueError(f"Not enough data for scenario: {scenario}")
    
    # Calculate returns for the period
    start_prices = period_prices.iloc[0]
    end_prices = period_prices.iloc[-1]
    period_returns = (end_prices / start_prices) - 1
    
    # Calculate portfolio impact
    portfolio_stocks = list(portfolio_weights.keys())
    available_stocks = [stock for stock in portfolio_stocks if stock in period_returns.index]
    
    if not available_stocks:
        raise ValueError(f"No portfolio stocks have data for scenario: {scenario}")
    
    # Calculate expected returns for available stocks
    stock_impacts = {}
    for stock in available_stocks:
        stock_impacts[stock] = period_returns[stock]
    
    # For missing stocks, use average or proxy
    missing_stocks = [stock for stock in portfolio_stocks if stock not in available_stocks]
    avg_return = period_returns.mean()
    
    for stock in missing_stocks:
        stock_impacts[stock] = avg_return
    
    # Calculate portfolio impact
    portfolio_impact = sum(portfolio_weights[stock] * stock_impacts[stock] for stock in portfolio_stocks)
    
    # Prepare result
    result = {
        'scenario': scenario,
        'start_date': start_date,
        'end_date': end_date,
        'stock_impacts': stock_impacts,
        'portfolio_impact': portfolio_impact,
    }
    
    # If market benchmark is available, add it
    if '^GSPC' in period_returns.index:
        result['market_impact'] = period_returns['^GSPC']
    
    return result

def plot_stress_test_results(stress_results):
    """
    Create plots for stress test results
    
    Parameters:
    - stress_results: Dictionary with stress test results
    
    Returns:
    - Matplotlib figure for stress test visualization
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Extract data
    scenario = stress_results['scenario']
    stock_impacts = stress_results['stock_impacts']
    portfolio_impact = stress_results['portfolio_impact']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar chart of individual stock impacts
    stocks = list(stock_impacts.keys())
    impacts = list(stock_impacts.values())
    
    # Sort for better visualization
    sorted_data = sorted(zip(stocks, impacts), key=lambda x: x[1])
    sorted_stocks = [x[0] for x in sorted_data]
    sorted_impacts = [x[1] * 100 for x in sorted_data]  # Convert to percentage
    
    # Choose colors based on return (red for negative, green for positive)
    colors = ['green' if x >= 0 else 'red' for x in sorted_impacts]
    
    ax1.barh(sorted_stocks, sorted_impacts, color=colors, alpha=0.7)
    ax1.set_xlabel('Price Change (%)')
    ax1.set_title(f'Stock Performance in {scenario.replace("_", " ").title()} Scenario')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(sorted_impacts):
        ax1.text(v + np.sign(v) * 1, i, f'{v:.1f}%', va='center')
    
    # 2. Portfolio vs Market impact
    labels = ['Portfolio']
    values = [portfolio_impact * 100]  # Convert to percentage
    
    # Add market benchmark if available
    if 'market_impact' in stress_results:
        labels.append('Market (S&P 500)')
        values.append(stress_results['market_impact'] * 100)
    
    # Choose colors
    colors = ['green' if x >= 0 else 'red' for x in values]
    
    ax2.bar(labels, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Price Change (%)')
    ax2.set_title(f'Portfolio Impact in {scenario.replace("_", " ").title()} Scenario')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        ax2.text(i, v + np.sign(v) * 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    
    return fig

def run_backtest_with_benchmark(stock_data, portfolio_weights, start_date, end_date, initial_investment, rebalance_frequency, benchmark_type):
    """Helper function to run backtest with proper benchmark handling"""
    try:
        # Run backtest with specified benchmark
        backtest_results = backtest_portfolio(
            stock_data,
            portfolio_weights,
            start_date=start_date,
            end_date=end_date,
            initial_investment=initial_investment,
            rebalance_frequency=rebalance_frequency,
            benchmark_type=benchmark_type
        )
        
        return backtest_results
    except Exception as e:
        st.error(f"Backtest error: {str(e)}")
        return None

# Main application
def main():
    # Define stock universe (S&P 500 stocks + S&P 500 index)
    default_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
         'PG', 'NFLX', 'KO',
             
        '^GSPC'  # S&P 500 index
    ]
    
    # App state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'optimal_portfolio' not in st.session_state:
        st.session_state.optimal_portfolio = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'investment_amount' not in st.session_state:
        st.session_state.investment_amount = 10000
    
    # App title
    st.title("Portfolio Optimization")
    st.subheader("Build and optimize your investment portfolio")
    
    # Initialize
    with st.sidebar:
        st.header("Configuration")
        
        # Date range selection
        st.subheader("Time Period")
        end_date = datetime.now()
        years_back = st.slider("Years of historical data", min_value=1, max_value=10, value=5)
        start_date = end_date - timedelta(days=years_back*365)
        
        st.write(f"**Date Range:**")
        st.write(f"- Start: {start_date.strftime('%Y-%m-%d')}")
        st.write(f"- End: {end_date.strftime('%Y-%m-%d')}")
        
        # Add investment amount input in sidebar
        st.subheader("Investment Amount")
        investment_amount = st.number_input(
            "Enter your investment amount ($)",
            min_value=1000,
            max_value=10000000,
            value=st.session_state.investment_amount,
            step=1000,
            format="%d"
        )
        st.session_state.investment_amount = investment_amount
        
        # ML toggle
        st.subheader("Machine Learning")
        use_ml = st.checkbox("Use Machine Learning", value=True, 
                          help="Enable to use predictions to enhance portfolio optimization")
        
        # Data loading button
        load_data = st.button("Load Stock Data")
    
    # Load data if button is pressed
    if load_data or st.session_state.stock_data is not None:
        # Show loading spinner while fetching data
        if st.session_state.stock_data is None:
            with st.spinner("Loading stock data..."):
                st.session_state.stock_data = get_stock_data(default_tickers, start_date, end_date)
                if st.session_state.stock_data is None:
                    st.error("Failed to download stock data. Please check your internet connection and try again.")
                    return
        
        # If data is loaded successfully
        st.success("Stock data loaded successfully!")
        
        # Main app tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Optimizer", "Stock Performance", "Machine Learning Insights", "Portfolio Backtest"])
        
        with tab1:
            st.header("Portfolio Optimizer")
            
            # Stock selection
            stocks_without_sp500 = [ticker for ticker in default_tickers if ticker != '^GSPC']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_stocks = st.multiselect(
                    "Select stocks for your portfolio (5-10 recommended)",
                    options=stocks_without_sp500,
                    default=stocks_without_sp500[:6]
                )
            
            with col2:
                risk_preference = st.radio(
                    "Select your risk preference",
                    options=["Conservative", "Moderate", "Aggressive"],
                    index=1
                ).lower()
            
            # Minimum 2 stocks required
            if len(selected_stocks) < 2:
                st.warning("Please select at least 2 stocks for your portfolio.")
            else:
                # Optimize portfolio button
                optimize_button = st.button("Optimize Portfolio")
                
                if optimize_button or st.session_state.optimal_portfolio is not None:
                    # Only run optimization if button is pressed or portfolio exists
                    if optimize_button:
                        with st.spinner("Optimizing your portfolio..."):
                            st.session_state.optimal_portfolio = create_optimized_portfolio(
                                selected_stocks, 
                                st.session_state.stock_data, 
                                risk_preference,
                                use_ml  # Pass the ML toggle value
                            )
                    
                    # Display results
                    if st.session_state.optimal_portfolio:
                        portfolio_result = st.session_state.optimal_portfolio
                        
                        # Display metrics
                        st.subheader("Portfolio Overview")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Expected Annual Return", f"{portfolio_result['expected_annual_return']*100:.2f}%")
                        col2.metric("Expected Volatility", f"{portfolio_result['expected_volatility']*100:.2f}%")
                        col3.metric("Sharpe Ratio", f"{portfolio_result['expected_sharpe_ratio']:.2f}")
                        
                        # Display allocation
                        st.subheader("Portfolio Allocation")
                        
                        # Use the user-defined investment amount instead of fixed $10,000
                        investment_amount = st.session_state.investment_amount
                        
                        allocation_data = []
                        for idx, (stock, weight) in enumerate(sorted(portfolio_result['portfolio_weights'].items(), key=lambda x: x[1], reverse=True), 1):
                            allocation_data.append({
                                'Stock': stock,
                                'Weight (%)': f"{weight*100:.2f}%",
                                f'Allocation (${investment_amount:,})': f"${weight*investment_amount:.2f}",
                                'Expected Return (%)': f"{portfolio_result['stock_returns'][stock]*100:.2f}%"
                            })

                        allocation_df = pd.DataFrame(allocation_data)
                        # Set the index to start from 1 instead of 0
                        allocation_df.index = range(1, len(allocation_df) + 1)
                        st.dataframe(allocation_df, use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Portfolio Allocation")
                            pie_chart = plot_portfolio_allocation(portfolio_result['portfolio_weights'])
                            st.pyplot(pie_chart)
                        
                        with col2:
                            st.subheader("Efficient Frontier")
                            ef_chart = plot_efficient_frontier(
                                portfolio_result['stock_returns'], 
                                st.session_state.stock_data, 
                                selected_stocks, 
                                portfolio_result
                            )
                            st.pyplot(ef_chart)
        
        with tab2:
            st.header("Stock Performance Analysis")
            
            if 'stock_data' in st.session_state and st.session_state.stock_data is not None:
                # Time period selection
                time_period = st.slider(
                    "Select analysis period (days)",
                    min_value=30,
                    max_value=365*years_back,
                    value=365,
                    step=30
                )
                
                # Stock selection for analysis
                analysis_stocks = st.multiselect(
                    "Select stocks to analyze",
                    options=stocks_without_sp500,
                    default=stocks_without_sp500[:5] if not selected_stocks else selected_stocks[:5]
                )
                
                if analysis_stocks:
                    # Stock performance comparison
                    st.subheader("Stock Price Comparison")
                    perf_chart = plot_stock_comparison(st.session_state.stock_data, analysis_stocks, time_period)
                    st.pyplot(perf_chart)
                    
                    # Stock returns and statistics
                    st.subheader("Stock Returns Analysis")
                    
                    # Calculate key statistics
                    prices = st.session_state.stock_data['Close'][analysis_stocks].tail(time_period)
                    returns = prices.pct_change().dropna()
                    
                    # Create statistics table
                    stats = {
                        'Mean Daily Return (%)': returns.mean() * 100,
                        'Standard Deviation (%)': returns.std() * 100,
                        'Minimum Daily Return (%)': returns.min() * 100,
                        'Maximum Daily Return (%)': returns.max() * 100,
                        'Annualized Return (%)': returns.mean() * 252 * 100,
                        'Annualized Volatility (%)': returns.std() * np.sqrt(252) * 100,
                        'Cumulative Return (%)': ((1 + returns).cumprod().iloc[-1] - 1) * 100
                    }
                    
                    stats_df = pd.DataFrame(stats).transpose()
                    st.dataframe(stats_df, use_container_width=True)
                    
                    
                    
                    # Risk vs Return
                    st.subheader("Risk vs Return")
                    if st.session_state.optimal_portfolio and 'stock_returns' in st.session_state.optimal_portfolio:
                        risk_return_chart = plot_return_vs_risk(
                            st.session_state.optimal_portfolio['stock_returns'], 
                            analysis_stocks
                        )
                    else:
                        # Calculate annualized returns and volatility
                        mean_returns = returns.mean() * 252
                        risk_return_chart = plot_return_vs_risk(mean_returns, analysis_stocks)
                    
                    if risk_return_chart:  # Only display if not None
                        st.pyplot(risk_return_chart)
                    else:
                        st.warning("Unable to create risk vs return chart with the selected stocks.")
        
        with tab3:
            st.header("Machine Learning Insights")
            
            st.info("This section shows how machine learning is used to enhance portfolio optimization.")
            
            if st.button("Run Machine Learning Analysis"):
                with st.spinner("Training machine learning models. This may take a while..."):
                    # Calculate portfolio metrics
                    metrics = calculate_portfolio_metrics(st.session_state.stock_data)
                    
                    # Prepare features
                    X, y = prepare_features(metrics)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models = train_models(X_train_scaled, y_train)
                    
                    # Evaluate models
                    results = []
                    for name, model in models.items():
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # Store results
                        results.append({
                            'Model': name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        })
                    
                    # Create a DataFrame with all results
                    results_df = pd.DataFrame(results)
                    
                    # Display model comparison
                    st.subheader("Model Comparison")
                    st.dataframe(results_df.sort_values('Accuracy', ascending=False), use_container_width=True)
                    
                    # Create a bar chart for model comparison
                    plt.close('all')
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111)
                    
                    # Sort by accuracy
                    sorted_df = results_df.sort_values('Accuracy', ascending=False)
                    
                    # Create a color palette
                    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
                    
                    # Plot the bars
                    bar_width = 0.2
                    positions = np.arange(len(sorted_df))
                    
                    ax.bar(positions - bar_width*1.5, sorted_df['Accuracy'], bar_width, label='Accuracy', color=colors[0])
                    ax.bar(positions - bar_width*0.5, sorted_df['Precision'], bar_width, label='Precision', color=colors[1])
                    ax.bar(positions + bar_width*0.5, sorted_df['Recall'], bar_width, label='Recall', color=colors[2])
                    ax.bar(positions + bar_width*1.5, sorted_df['F1-Score'], bar_width, label='F1-Score', color=colors[3])
                    
                    # Add labels and title
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Score')
                    ax.set_title('Model Performance Comparison')
                    ax.set_xticks(positions)
                    ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Get feature importance from Random Forest
                    if 'Random Forest' in models:
                        rf_model = models['Random Forest']
                        feature_importance = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.subheader("Feature Importance (Random Forest)")
                        
                        # Display top features
                        top_features = feature_importance.head(20)
                        
                        # Create bar chart
                        plt.close('all')
                        fig = plt.figure(figsize=(10, 8))
                        ax = fig.add_subplot(111)
                        
                        # Use a color gradient for feature importance
                        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_features)))
                        
                        bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
                        
                        # Add feature names
                        ax.set_yticks(range(len(top_features)))
                        ax.set_yticklabels(top_features['Feature'])
                        
                        # Add value labels
                        for i, v in enumerate(top_features['Importance']):
                            ax.text(v + 0.005, i, f'{v:.4f}', va='center')
                        
                        ax.set_title('Top 20 Feature Importance (Random Forest)')
                        ax.set_xlabel('Importance')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Display table of feature importance
                        st.dataframe(top_features, use_container_width=True)
                    
                    st.success("Machine learning analysis complete!")
            
            # Explanation of ML in portfolio optimization
            st.subheader("How Machine Learning Enhances Portfolio Optimization")
            st.markdown("""
            The portfolio optimization process uses machine learning in several ways:
            
            1. **Stock Performance Prediction**: ML models predict which stocks may outperform the market in the future.
            
            2. **Feature Importance**: The models identify which financial metrics best predict future performance.
            
            3. **Risk Assessment**: Machine learning helps assess the risk associated with different stocks and portfolios.
            
            4. **Dynamic Allocation**: The system can adjust portfolio weights based on changing market conditions.
            
            The Random Forest model is particularly effective for this purpose due to its ability to capture complex 
            relationships in financial data without overfitting.
            """)

        

        with tab4:
            st.header("Portfolio Backtest")
            
            # Create subtabs for different backtest features
            backtest_subtabs = st.tabs(["Basic Backtest", "Strategy Comparison", "Stress Testing"])
            
            with backtest_subtabs[0]:  # Basic Backtest
                if 'optimal_portfolio' not in st.session_state or st.session_state.optimal_portfolio is None:
                    st.warning("Please optimize a portfolio first in the 'Portfolio Optimizer' tab.")
                else:
                    # Setup backtest parameters
                    st.subheader("Backtest Configuration")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Date range for backtest
                        end_date = datetime.now().date()
                        start_date_years = st.slider(
                            "Backtest period (years)",
                            min_value=1,
                            max_value=10,
                            value=3
                        )
                        start_date = end_date - timedelta(days=365*start_date_years)
                        
                        st.write(f"**Backtest Period:** {start_date} to {end_date}")
                        
                        # Initial investment
                        initial_investment = st.number_input(
                            "Initial Investment ($)",
                            min_value=1000,
                            max_value=10000000,
                            value=10000,
                            step=1000
                        )
                    
                    with col2:
                        # Rebalancing frequency
                        rebalance_frequency = st.selectbox(
                            "Rebalancing Frequency",
                            options=["never", "monthly", "quarterly", "yearly"],
                            index=1
                        )
                        
                        # Benchmark selection
                        benchmark = st.selectbox(
                            "Benchmark",
                            options=["S&P 500", "Equal Weight"],
                            index=0
                        )
                        
                        # Download button for backtest results
                        include_benchmark = st.checkbox("Include benchmark in results", value=True)
                    
                    # Run backtest button
                    if st.button("Run Backtest"):
                            with st.spinner("Running portfolio backtest..."):
                                try:
                                    # Get portfolio weights from the optimizer
                                    portfolio_weights = st.session_state.optimal_portfolio['portfolio_weights']
                                    
                                    # Run backtest with selected benchmark
                                    backtest_results = run_backtest_with_benchmark(
                                        st.session_state.stock_data,
                                        portfolio_weights,
                                        start_date=start_date,
                                        end_date=end_date,
                                        initial_investment=initial_investment,
                                        rebalance_frequency=rebalance_frequency,
                                        benchmark_type=benchmark  # Use the selected benchmark
                                    )
                                    
                                    if backtest_results:
                                        # Store in session state
                                        st.session_state.backtest_results = backtest_results
                                        
                                        # Display backtest results
                                        st.subheader("Backtest Performance")
                                        
                                        # Show metrics
                                        metrics_df = create_backtest_metrics_table(backtest_results)
                                        st.dataframe(metrics_df, use_container_width=True)
                                        
                                        # Performance charts
                                        st.subheader("Performance Charts")
                                        
                                        # Value over time chart
                                        plots = plot_backtest_performance(backtest_results)
                                        
                                        # Main chart - Portfolio Value
                                        st.pyplot(plots['value_plot'])
                                        
                                        # Relative performance chart (if benchmark is available)
                                        if 'relative_performance' in plots:
                                            st.subheader(f"Relative Performance vs {benchmark}")
                                            st.pyplot(plots['relative_performance'])
                                        
                                        # Drawdown chart
                                        st.subheader("Drawdown Analysis")
                                        st.pyplot(plots['drawdown_plot'])
                                        
                                        # Monthly returns heatmap
                                        if 'monthly_heatmap' in plots:
                                            st.subheader("Monthly Returns Heatmap")
                                            st.pyplot(plots['monthly_heatmap'])
                                except Exception as e:
                                    st.error(f"Backtest error: {str(e)}")
                                    st.error(f"Please check your portfolio and try again.")
            
            
            with backtest_subtabs[1]:  # Strategy Comparison
                st.subheader("Strategy Comparison")
                
                # Explain the feature
                st.markdown("""
                Compare your optimized portfolio against different portfolio strategies:
                - **Conservative Strategy**: Focuses on minimizing volatility
                - **Moderate Strategy**: Balances return and risk (maximizes Sharpe ratio)
                - **Aggressive Strategy**: Prioritizes expected returns
                """)
                
                if 'stock_data' not in st.session_state or st.session_state.stock_data is None:
                    st.warning("Please load stock data first.")
                else:
                    # Get all available stocks
                    if isinstance(st.session_state.stock_data.columns, pd.MultiIndex):
                        all_stocks = [col for col in st.session_state.stock_data['Close'].columns if col != '^GSPC']
                    else:
                        all_stocks = [col for col in st.session_state.stock_data.columns if col != '^GSPC']
                    
                    # If we have an optimized portfolio, use those stocks as default
                    default_stocks = []
                    if 'optimal_portfolio' in st.session_state and st.session_state.optimal_portfolio is not None:
                        default_stocks = list(st.session_state.optimal_portfolio['portfolio_weights'].keys())
                    
                    if not default_stocks:
                        default_stocks = all_stocks[:5]
                    
                    # Stock selection for comparison
                    selected_stocks = st.multiselect(
                        "Select stocks for strategy comparison",
                        options=all_stocks,
                        default=default_stocks
                    )
                    
                    if len(selected_stocks) >= 2:
                        # Backtest parameters
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Date range for backtest
                            end_date = datetime.now().date()
                            start_date_years = st.slider(
                                "Comparison period (years)",
                                min_value=1,
                                max_value=10,
                                value=3,
                                key="compare_years"
                            )
                            start_date = end_date - timedelta(days=365*start_date_years)
                            
                            # Initial investment
                            initial_investment = st.number_input(
                                "Initial Investment ($)",
                                min_value=1000,
                                max_value=10000000,
                                value=10000,
                                step=1000,
                                key="compare_investment"
                            )
                        
                        with col2:
                            # Select strategies to compare
                            strategies_to_compare = st.multiselect(
                                "Select strategies to compare",
                                options=["Optimal Portfolio", "Conservative", "Moderate", "Aggressive", "Equal Weight"],
                                default=["Optimal Portfolio", "Conservative", "Aggressive"]
                            )
                        
                        # Run comparison button
                        if st.button("Compare Strategies"):
                            with st.spinner("Comparing investment strategies..."):
                                try:
                                    # Build strategies dictionary
                                    strategies = {}
                                    
                                    # Add optimal portfolio if selected and available
                                    if "Optimal Portfolio" in strategies_to_compare:
                                        if 'optimal_portfolio' in st.session_state and st.session_state.optimal_portfolio is not None:
                                            strategies["Optimal Portfolio"] = st.session_state.optimal_portfolio['portfolio_weights']
                                        else:
                                            st.warning("Optimal Portfolio not available. Please optimize a portfolio first.")
                                    
                                    # Add predefined strategies
                                    if "Conservative" in strategies_to_compare:
                                        strategies["Conservative"] = generate_risk_strategy(
                                            st.session_state.stock_data, selected_stocks, "low"
                                        )
                                    
                                    if "Moderate" in strategies_to_compare:
                                        strategies["Moderate"] = generate_risk_strategy(
                                            st.session_state.stock_data, selected_stocks, "moderate"
                                        )
                                    
                                    if "Aggressive" in strategies_to_compare:
                                        strategies["Aggressive"] = generate_risk_strategy(
                                            st.session_state.stock_data, selected_stocks, "high"
                                        )
                                    
                                    if "Equal Weight" in strategies_to_compare:
                                        strategies["Equal Weight"] = {stock: 1/len(selected_stocks) for stock in selected_stocks}
                                    
                                    # Run comparison if we have at least one strategy
                                    if strategies:
                                        comparison_results = compare_strategies(
                                            st.session_state.stock_data,
                                            strategies,
                                            start_date=start_date,
                                            end_date=end_date,
                                            initial_investment=initial_investment
                                        )
                                        
                                        # Store in session state
                                        st.session_state.strategy_comparison = comparison_results
                                        
                                        # Plot comparison results
                                        comparison_plots = plot_strategy_comparison(comparison_results)
                                        
                                        # Show performance comparison
                                        st.subheader("Performance Comparison")
                                        st.pyplot(comparison_plots['value_comparison'])
                                        
                                        # Show drawdown comparison
                                        st.subheader("Drawdown Comparison")
                                        st.pyplot(comparison_plots['drawdown_comparison'])
                                        
                                        # Show metrics comparison
                                        st.subheader("Strategy Metrics Comparison")
                                        st.dataframe(comparison_plots['metrics_table'], use_container_width=True)
                                        st.pyplot(comparison_plots['metrics_comparison'])
                                        
                                        # Strategy breakdown - show weights of each strategy
                                        st.subheader("Strategy Allocations")
                                        
                                        for name, weights in strategies.items():
                                            with st.expander(f"{name} Strategy Weights"):
                                                # Sort weights by value (descending)
                                                sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
                                                
                                                # Create DataFrame for display
                                                weights_df = pd.DataFrame({
                                                    'Stock': list(sorted_weights.keys()),
                                                    'Weight (%)': [f"{v*100:.2f}%" for v in sorted_weights.values()]
                                                })

                                                # Set the index to start from 1 instead of 0
                                                weights_df.index = range(1, len(weights_df) + 1)
                                                
                                                st.dataframe(weights_df, use_container_width=True)
                                    else:
                                        st.error("No strategies selected for comparison.")
                                    
                                except Exception as e:
                                    st.error(f"Strategy comparison error: {str(e)}")
                    else:
                        st.warning("Please select at least 2 stocks for strategy comparison.")
            
            with backtest_subtabs[2]:  # Stress Testing
                st.subheader("Portfolio Stress Testing")
                
                st.markdown("""
                This tool tests how your optimized portfolio would perform under different market stress scenarios:
                - **COVID-19 Crash (2020)**: The rapid market collapse during the early pandemic
                - **Inflation Surge (2022)**: The market reaction to rising inflation
                """)
                
                if 'optimal_portfolio' not in st.session_state or st.session_state.optimal_portfolio is None:
                    st.warning("Please optimize a portfolio first in the 'Portfolio Optimizer' tab.")
                else:
                    # Get portfolio weights
                    portfolio_weights = st.session_state.optimal_portfolio['portfolio_weights']
                    
                    # Select stress scenario
                    scenario = st.selectbox(
                        "Select stress scenario",
                        options=[
                             
                            "covid_crash", 
                             
                            "inflation_2022"
                        ],
                        format_func=lambda x: {
                            
                            "covid_crash": "COVID-19 Crash (2020)",
                            
                            "inflation_2022": "Inflation Surge (2022)"
                        }.get(x, x)
                    )
                    
                    # Run stress test button
                    if st.button("Run Stress Test"):
                        with st.spinner("Running portfolio stress test..."):
                            try:
                                # Perform stress test
                                stress_results = perform_stress_test(
                                    portfolio_weights,
                                    st.session_state.stock_data,
                                    scenario
                                )
                                
                                # Display results
                                st.subheader("Stress Test Results")
                                
                                # Show overall impact
                                impact = stress_results['portfolio_impact'] * 100  # Convert to percentage
                                impact_color = "green" if impact >= 0 else "red"
                                
                                st.markdown(f"""
                                **Scenario**: {scenario.replace('_', ' ').title()}  
                                **Period**: {stress_results['start_date']} to {stress_results['end_date']}  
                                **Portfolio Impact**: <span style='color:{impact_color}'>{impact:.2f}%</span>
                                """, unsafe_allow_html=True)
                                
                                # Show market impact if available
                                if 'market_impact' in stress_results:
                                    market_impact = stress_results['market_impact'] * 100  # Convert to percentage
                                    market_color = "green" if market_impact >= 0 else "red"
                                    st.markdown(f"**S&P 500 Impact**: <span style='color:{market_color}'>{market_impact:.2f}%</span>", unsafe_allow_html=True)
                                    
                                    # Compare to market
                                    relative_impact = impact - market_impact
                                    relative_color = "green" if relative_impact >= 0 else "red"
                                    st.markdown(f"**Relative Performance**: <span style='color:{relative_color}'>{relative_impact:.2f}%</span>", unsafe_allow_html=True)
                                
                                # Plot stress test visualization
                                st.pyplot(plot_stress_test_results(stress_results))
                                
                                # Display individual stock impacts
                                stock_impacts = stress_results['stock_impacts']
                                impact_df = pd.DataFrame({
                                    'Stock': list(stock_impacts.keys()),
                                    'Impact (%)': [f"{v*100:.2f}%" for v in stock_impacts.values()]
                                })
                                
                                # Sort by impact (descending)
                                impact_df['Impact (Numeric)'] = [v*100 for v in stock_impacts.values()]
                                impact_df = impact_df.sort_values('Impact (Numeric)', ascending=False).drop('Impact (Numeric)', axis=1)

                                # Set the index to start from 1 instead of 0
                                impact_df.index = range(1, len(impact_df) + 1)
                                
                                st.subheader("Individual Stock Impacts")
                                st.dataframe(impact_df, use_container_width=True)
                                
                                # Analysis and recommendations
                                st.subheader("Analysis & Portfolio Resilience")
                                
                                # Compare portfolio performance to market
                                if 'market_impact' in stress_results:
                                    if impact > market_impact:
                                        st.success(f"Your portfolio outperformed the S&P 500 by {relative_impact:.2f}% during this stress scenario.")
                                    else:
                                        st.warning(f"Your portfolio underperformed the S&P 500 by {abs(relative_impact):.2f}% during this stress scenario.")
                                
                                # Identify most resilient and vulnerable stocks
                                resilient_stocks = impact_df.iloc[:3]['Stock'].tolist() if len(impact_df) >= 3 else impact_df['Stock'].tolist()
                                vulnerable_stocks = impact_df.iloc[-3:]['Stock'].tolist() if len(impact_df) >= 3 else impact_df['Stock'].tolist()
                                
                                # Recommendations based on stress test
                                st.markdown("#### Portfolio Resilience Recommendations")
                                
                                if impact < -15:  # Significant negative impact
                                    st.markdown("""
                                    - **High Vulnerability**: Your portfolio shows significant vulnerability to this type of market stress
                                    - **Consider Hedging**: Adding defensive assets could improve resilience
                                    - **Diversification**: Increase exposure to less correlated assets
                                    """)
                                elif impact < -5:  # Moderate negative impact
                                    st.markdown("""
                                    - **Moderate Vulnerability**: Your portfolio shows some vulnerability to this type of market stress
                                    - **Rebalance**: Consider reducing allocation to most vulnerable stocks
                                    - **Sector Exposure**: Review sector allocation to reduce concentration risk
                                    """)
                                else:  # Low negative impact or positive
                                    st.markdown("""
                                    - **Strong Resilience**: Your portfolio shows good resilience to this type of market stress
                                    - **Maintain Strategy**: Current allocation appears well-positioned for similar scenarios
                                    - **Fine-tuning**: Consider increasing allocation to most resilient stocks
                                    """)
                                
                            except Exception as e:
                                st.error(f"Stress test error: {str(e)}")
                                st.info("Some historical scenarios may not have data for all your stocks. Try a different scenario or portfolio.")

        # Footer    

if __name__ == "__main__":
    main()