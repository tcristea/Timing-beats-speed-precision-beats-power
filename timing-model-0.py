# Below is a Python scaffolding for your timing model. It is designed to be modular, so you can replace components (e.g., signal generation, validation) as needed. # The code includes placeholders and comments to guide your implementation.
"""
Timing Model for a Fixed Portfolio (up to 25 stocks)
In-sample hypothesis generation -> out-of-sample validation -> adaptation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Data Module
# ============================================================

def load_data(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for each ticker. 
    Replace with your data source (e.g., yfinance, CSV files).
    """
    data = {}
    for ticker in tickers:
        # Example: using yfinance
        # df = yf.download(ticker, start=start, end=end)
        # data[ticker] = df[['Open','High','Low','Close','Volume']]
        pass
    return data

def prepare_features(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Compute technical indicators for each stock.
    Example: moving averages, RSI, volatility, volume ratios.
    """
    features = {}
    for ticker, df in data.items():
        df = df.copy()
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        df['rsi'] = compute_rsi(df['Close'], 14)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        # Drop NaN rows
        features[ticker] = df.dropna()
    return features

def compute_rsi(prices, period=14):
    """Simple RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================
# 2. Signal Generation
# ============================================================

@dataclass
class Signal:
    """Container for trading signals: +1 (long), -1 (short), 0 (neutral)"""
    date: pd.Timestamp
    ticker: str
    direction: int  # 1, -1, 0
    strength: float  # optional confidence

class BaseSignalModel:
    """Abstract base for signal generators"""
    def __init__(self):
        pass
    
    def fit(self, features: Dict[str, pd.DataFrame]) -> None:
        """Train the model on in-sample data"""
        raise NotImplementedError
    
    def predict(self, features: pd.DataFrame) -> int:
        """Generate signal for a single day for one ticker"""
        raise NotImplementedError
    
    def predict_all(self, features_dict: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate signals for all tickers on all dates"""
        signals = []
        for ticker, df in features_dict.items():
            for idx, row in df.iterrows():
                signal = self.predict(row.to_frame().T)  # pass a single row
                if signal != 0:
                    signals.append(Signal(date=idx, ticker=ticker, direction=signal))
        return signals

# Example: Simple moving average crossover model (hard-coded)
class MACrossover(BaseSignalModel):
    def __init__(self, fast=20, slow=50):
        self.fast = fast
        self.slow = slow
        self.feature_names = [f'ma_{fast}', f'ma_{slow}']
    
    def predict(self, features: pd.DataFrame) -> int:
        # features is a DataFrame with one row
        ma_fast = features[f'ma_{self.fast}'].values[0]
        ma_slow = features[f'ma_{self.slow}'].values[0]
        if ma_fast > ma_slow:
            return 1
        elif ma_fast < ma_slow:
            return -1
        else:
            return 0

# ============================================================
# 3. Backtesting Engine
# ============================================================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    ticker: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float

class Backtester:
    def __init__(self, initial_capital: float = 100000, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run(self, signals: List[Signal], price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Simulate trading based on signals.
        Returns a DataFrame with daily portfolio value and trade log.
        """
        # Sort signals by date
        signals = sorted(signals, key=lambda s: s.date)
        
        # For simplicity, assume we can trade at next day's close.
        # We'll merge signals with prices.
        
        # Build a DataFrame of all dates across all tickers
        all_dates = pd.date_range(start=min(s.date for s in signals), 
                                   end=max(s.date for s in signals))
        portfolio = pd.DataFrame(index=all_dates)
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0  # value of stocks held
        portfolio['total'] = self.initial_capital
        
        # Track positions: dict[ticker] = {entry_date, entry_price, quantity, direction}
        positions = {}
        trade_log = []
        
        for date in all_dates:
            # Process signals that occur on this date (assuming signal at market close)
            day_signals = [s for s in signals if s.date == date]
            # For each signal, we will execute at next day's open (simplified)
            # Here we just record intent; actual execution in next step.
            
            # First, update positions with today's prices (mark to market)
            for ticker, pos in list(positions.items()):
                if ticker in price_data:
                    today_price = price_data[ticker].loc[date, 'Close'] if date in price_data[ticker].index else np.nan
                    if pd.notna(today_price):
                        pos['current_price'] = today_price
                        pos['pnl'] = (today_price - pos['entry_price']) * pos['quantity'] * pos['direction']
            
            # Then, handle exits and entries based on signals (simplified: all at close)
            # In a real system you would model slippage and order timing.
            # For scaffolding, we skip detailed execution.
        
        return portfolio, trade_log

# ============================================================
# 4. Model Discovery (Genetic Programming Example)
# ============================================================

def discover_models_in_sample(features_dict: Dict[str, pd.DataFrame], 
                              population_size: int = 50, 
                              generations: int = 10) -> List[BaseSignalModel]:
    """
    Use genetic programming to evolve trading rules.
    Placeholder - you would implement using DEAP or similar.
    """
    # This function should:
    # - Define a grammar of indicators and operators
    # - Generate random rules
    # - Evaluate fitness (e.g., Sharpe ratio) on in-sample data (with cross-validation)
    # - Select, crossover, mutate
    # - Return top N rules
    top_rules = []
    # ...
    return top_rules

# ============================================================
# 5. Validation and Adaptation
# ============================================================

def validate_out_of_sample(model: BaseSignalModel, 
                           features_dict: Dict[str, pd.DataFrame], 
                           price_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Run backtest on out-of-sample data and return performance metrics.
    """
    signals = model.predict_all(features_dict)
    bt = Backtester()
    portfolio, trades = bt.run(signals, price_data)
    metrics = compute_metrics(portfolio, trades)
    return metrics

def compute_metrics(portfolio: pd.DataFrame, trades: List[Trade]) -> Dict:
    """Calculate Sharpe, max drawdown, win rate, etc."""
    # Placeholder
    return {}

def rolling_retrain(model_class, features_dict: Dict[str, pd.DataFrame], 
                    window_years: int = 2, step_months: int = 1) -> List[BaseSignalModel]:
    """
    Periodically retrain the model on the most recent data.
    Returns a list of models (one per window).
    """
    models = []
    # Determine date ranges
    all_dates = pd.DatetimeIndex(sorted(set.union(*[set(df.index) for df in features_dict.values()])))
    start_date = all_dates.min()
    end_date = all_dates.max()
    
    current_start = start_date
    while current_start + pd.DateOffset(years=window_years) <= end_date:
        window_end = current_start + pd.DateOffset(years=window_years)
        # Slice features_dict for this window
        window_features = {}
        for ticker, df in features_dict.items():
            window_features[ticker] = df.loc[current_start:window_end].copy()
        # Train model on this window
        model = model_class()
        model.fit(window_features)
        models.append(model)
        # Move window forward
        current_start += pd.DateOffset(months=step_months)
    return models

# ============================================================
# 6. Main Pipeline
# ============================================================

if __name__ == "__main__":
    # Configuration
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # up to 25
    START_IS = '2010-01-01'
    END_IS = '2019-12-31'
    START_OOS = '2020-01-01'
    END_OOS = '2023-12-31'
    
    # 1. Load and prepare data
    raw_data = load_data(TICKERS, START_IS, END_OOS)  # load full range
    features_all = prepare_features(raw_data)
    
    # Split into in-sample and out-of-sample
    features_is = {}
    features_oos = {}
    for ticker, df in features_all.items():
        features_is[ticker] = df.loc[START_IS:END_IS]
        features_oos[ticker] = df.loc[START_OOS:END_OOS]
    
    # 2. Discover models on in-sample data
    print("Discovering models in-sample...")
    discovered_models = discover_models_in_sample(features_is, population_size=100, generations=20)
    
    # 3. Validate each discovered model out-of-sample
    print("Validating out-of-sample...")
    validated_models = []
    for model in discovered_models:
        metrics = validate_out_of_sample(model, features_oos, raw_data)
        if metrics.get('sharpe', -10) > 1.0:  # example threshold
            validated_models.append(model)
    
    # 4. Adaptation: rolling retrain on most recent data (e.g., last 2 years)
    print("Setting up rolling retraining...")
    # Use the best model class (or ensemble) for adaptation
    best_model_class = type(validated_models[0]) if validated_models else MACrossover
    rolling_models = rolling_retrain(best_model_class, features_all, window_years=2, step_months=1)
    
    # 5. (Optional) Synthetic data robustness test
    # generate_synthetic_paths() and test models
