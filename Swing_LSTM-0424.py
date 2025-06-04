import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replaced 'import ta'
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Prompt for user input
ticker = input("Enter stock symbol (e.g., NVDA): ").upper()
funds = float(input("Enter funds to invest (e.g., 10000): "))

# Fetch data
try:
    data = yf.download(ticker, start='2018-01-01', end='2025-04-24', progress=False)
    vix = yf.download('^VIX', start='2018-01-01', end='2025-04-24', progress=False)['Close'].squeeze()
    # Debug: Check data
    print("VIX type (yfinance):", type(vix))
    print("VIX data (yfinance):\n", vix.head())
    print("Null values in VIX (yfinance):", vix.isnull().sum())
except Exception as e:
    print(f"Error fetching yfinance data: {e}. Trying CBOE VIX data...")
    try:
        vix = pd.read_csv('https://www.cboe.com/indices/vix_historical.csv', index_col='Date', parse_dates=True)['Close'].squeeze()
        vix = vix[vix.index >= '2018-01-01']
        data = yf.download(ticker, start='2018-01-01', end='2025-04-24', progress=False)
        # Debug: Check data
        print("VIX type (CBOE):", type(vix))
        print("VIX data (CBOE):\n", vix.head())
        print("Null values in VIX (CBOE):", vix.isnull().sum())
    except Exception as e:
        print(f"Error fetching CBOE data: {e}. Using SPY ATR as proxy.")
        spy = yf.download('SPY', start='2018-01-01', end='2025-04-24', progress=False)
        vix = ta.atr(spy['High'], spy['Low'], spy['Close'], length=14) * 100  # Scaled ATR
        # Handle initial nulls in ATR (first 14 days)
        vix = vix.fillna(method='bfill')  # Backward fill to handle initial nulls
        data = yf.download(ticker, start='2018-01-01', end='2025-04-24', progress=False)
        # Debug: Check data
        print("VIX type (ATR):", type(vix))
        print("VIX data (ATR):\n", vix.head())
        print("Null values in VIX (ATR):", vix.isnull().sum())

if data.empty:
    print(f"Error: No data for {ticker}. Check symbol or try again.")
    exit()
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Check data quality
if data.isnull().any().any():
    print("Data contains null values. Filling with forward fill...")
    data.fillna(method='ffill', inplace=True)

# Check VIX quality
if isinstance(vix, pd.Series) and vix.isnull().any():
    print("VIX data contains null values. Filling with forward fill...")
    vix = vix.fillna(method='ffill')
    print(f"Warning: Filled {vix.isnull().sum()} missing VIX values.")
elif isinstance(vix, pd.DataFrame) and vix.isnull().any().any():
    print("VIX DataFrame contains null values. Filling with forward fill...")
    vix = vix.fillna(method='ffill')
    print(f"Warning: Filled {vix.isnull().sum().sum()} missing VIX values.")
else:
    print("No null values in VIX data.")

# Calculate indicators
data['EMA13'] = ta.ema(data['Close'], length=13)
data['EMA21'] = ta.ema(data['Close'], length=21)
data['SMA100'] = ta.sma(data['Close'], length=100)
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
data['Stoch_K'] = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3, smooth_k=3)['STOCHk_14_3_3']
data['Stoch_D'] = ta.stoch(data['High'], data['Low'], data['Close'], k=14, d=3, smooth_k=3)['STOCHd_14_3_3']
data['OBV'] = ta.obv(data['Close'], data['Volume'])
data['Returns'] = data['Close'].pct_change()
data['Lag1_Return'] = data['Returns'].shift(1)
data['VIX'] = vix

# Fibonacci Retracement
def calculate_fibonacci_levels(data, lookback=20):
    fib_levels = []
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    for i in range(lookback, len(data)):
        window = data['Close'].iloc[i-lookback:i]
        high = window.max()
        low = window.min()
        range_ = high - low
        levels = [high - ratio * range_ for ratio in fib_ratios]
        closest_level = min(levels, key=lambda x: abs(x - data['Close'].iloc[i]))
        fib_level = (data['Close'].iloc[i] - closest_level) / range_ if range_ != 0 else 0
        fib_support = 1 if abs(data['Close'].iloc[i] - levels[1]) / range_ < 0.01 or abs(data['Close'].iloc[i] - levels[3]) / range_ < 0.01 else 0
        fib_trend = 1 if data['Close'].iloc[i] > (high - 0.5 * range_) else -1 if data['Close'].iloc[i] < (high - 0.5 * range_) else 0
        fib_levels.append([fib_level, fib_support, fib_trend])
    return pd.DataFrame(fib_levels, columns=['Fib_Level', 'Fib_Support', 'Fib_Trend'], index=data.index[lookback:])

fib_data = calculate_fibonacci_levels(data)
data = data.join(fib_data)

# Feature engineering
data['EMA13_21_Cross'] = (data['EMA13'] > data['EMA21']).astype(int)
data['Price_Above_SMA100'] = (data['Close'] > data['SMA100']).astype(int)
data['Stoch_Cross'] = ((data['Stoch_K'] > data['Stoch_D']) & (data['Stoch_K'] < 20)).astype(int) - \
                      ((data['Stoch_K'] < data['Stoch_D']) & (data['Stoch_K'] > 80)).astype(int)
data['Stoch_Level'] = (data['Stoch_K'] < 20).astype(int) - (data['Stoch_K'] > 80).astype(int)
data['OBV_Trend'] = (data['OBV'] > data['OBV'].rolling(window=20).mean()).astype(int)
data['VIX_High'] = (data['VIX'] > 15).astype(int)  # Adjusted threshold

# Target: 1 if price rises > 2x ATR within 10 days
data['Future_Price'] = data['Close'].shift(-10)
data['Target'] = ((data['Future_Price'] - data['Close']) > 2 * data['ATR']).astype(int)

# Drop NaN
data = data.dropna()

# Features and target
features = ['EMA13_21_Cross', 'Price_Above_SMA100', 'Stoch_K', 'Stoch_D', 'Stoch_Cross', 'Stoch_Level', 'Fib_Level', 'Fib_Support', 'Fib_Trend', 'ATR', 'OBV_Trend', 'Lag1_Return', 'VIX_High']
# Check correlations
corr_matrix = np.corrcoef(data[features].T)
print("Feature Correlations:\n", pd.DataFrame(corr_matrix, index=features, columns=features))
if corr_matrix[features.index('Stoch_K'), features.index('Stoch_D')] > 0.8:
    print("Warning: Stoch_K and Stoch_D highly correlated. Dropping Stoch_D.")
    features.remove('Stoch_D')

X = data[features].values
y = data['Target'].values

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Prepare LSTM input
lookback = 20
X_lstm, y_lstm = [], []
for i in range(lookback, len(X_scaled)):
    X_lstm.append(X_scaled[i-lookback:i])
    y_lstm.append(y[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Data augmentation
X_lstm += np.random.normal(0, 0.01, X_lstm.shape)

# Train-test split
train_size = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:train_size], X_lstm[train_size:]
y_train, y_test = y_lstm[:train_size], y_lstm[train_size:]

# Address class imbalance
print(f"Buy Signal Proportion: {np.mean(y):.2%}")
smote = SMOTE(random_state=42)
X_train_reshaped = X_train.reshape(-1, lookback * X_lstm.shape[2])
X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)
X_train = X_train_smote.reshape(-1, lookback, X_lstm.shape[2])

# Build LSTM
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(lookback, X_lstm.shape[2]), kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(LSTM(100, kernel_regularizer='l2'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train
class_weight = {0: 1.0, 1: 3.0}
model.fit(X_train, y_train_smote, epochs=100, batch_size=16, validation_data=(X_test, y_test), 
         callbacks=[early_stopping], class_weight=class_weight, verbose=0)

# Predict signals
predictions = model.predict(X_test, verbose=0)
signals = (predictions > 0.5).astype(int)

# Calculate win rate
win_rate = np.mean(signals.flatten() == y_test)

# Latest signal
latest_data = X_scaled[-lookback:].reshape(1, lookback, X_lstm.shape[2])
latest_signal = model.predict(latest_data, verbose=0)[0][0]
entry_price = data['Close'].iloc[-1]
atr = data['ATR'].iloc[-1]
stop_loss = entry_price - 1.5 * atr
profit_target = entry_price + 2 * atr
trailing_stop = 1.5 * atr
risk_per_trade = 500
position_size = min(risk_per_trade / (1.5 * atr), funds / entry_price)
position_size = int(position_size)
investment_used = position_size * entry_price

# Simulate trades
capital = funds
trades = []
returns_series = []
for i in range(len(signals)):
    if signals[i] == 1 and data['VIX_High'].iloc[train_size + i] == 1:
        atr_i = data['ATR'].iloc[train_size + i]
        stop_loss_i = 1.5 * atr_i
        pos_size = min(risk_per_trade / stop_loss_i, capital / data['Close'].iloc[train_size + i])
        pos_size = int(pos_size)
        if pos_size > 0:
            profit = 2 * atr_i if y_test[i] == 1 else -stop_loss_i
            trade_return = profit * pos_size
            trades.append(trade_return)
            returns_series.append(trade_return / capital)
            capital += trade_return
returns = np.sum(trades) / funds if trades else 0
max_drawdown = np.max(np.maximum.accumulate(returns_series) - returns_series) if returns_series else 0
sharpe = np.mean(returns_series) / np.std(returns_series) * np.sqrt(252) if returns_series else 0

# Output
print(f"\nTrade Analysis for {ticker}:")
print(f"Test Win Rate: {win_rate:.2%}")
print(f"Buy Signal Proportion: {np.mean(y_test):.2%}")
print(f"Number of Trades: {len(trades)}")
if latest_signal > 0.5 and data['VIX_High'].iloc[-1] == 1:
    print(f"Buy Signal: Buy {ticker} with {latest_signal:.2%} confidence")
else:
    print(f"No Buy Signal: {ticker} has {latest_signal:.2%} confidence or low VIX")
print(f"Entry Price: ${entry_price:.2f}")
print(f"Trailing Stop: ${trailing_stop:.2f} below highest price (initially ${stop_loss:.2f})")
print(f"Profit Target: ${profit_target:.2f} (optional)")
print(f"Position Size: {position_size} shares")
print(f"Investment Used: ${investment_used:.2f} (of ${funds:.2f} available)")
print(f"Risk per Trade: ${risk_per_trade:.2f}")
print(f"Potential Profit: Dynamic (e.g., ${(entry_price + 4 * atr - entry_price) * position_size:.2f} at 4x ATR)")
print(f"Simulated Return: {returns:.2%}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
