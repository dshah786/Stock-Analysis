import yfinance as yf
import pandas as pd
import numpy as np

class ElliottWaveAnalyzer:
    def __init__(self, ticker: str, period: str = '1y', interval: str = '1d'):
        """
        Initialize Elliott Wave Analyzer
        
        :param ticker: Stock ticker symbol
        :param period: Historical data period
        :param interval: Data interval
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = self._fetch_stock_data()
    
    def _fetch_stock_data(self) -> pd.DataFrame:
        """Fetch and preprocess historical stock data."""
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=self.period, interval=self.interval)
        
        if df.empty:
            raise ValueError(f"No data fetched for {self.ticker} with period='{self.period}' and interval='{self.interval}'.")
        
        print(f"Data fetched successfully for {self.ticker}. Number of rows: {len(df)}")
        df['Returns'] = df['Close'].pct_change()
        return df.dropna()
    
    def _detect_swings(self, prices: pd.Series, window: int = 5) -> (pd.Series, pd.Series):
        """Detect swing highs and lows."""
        swing_highs = (prices > prices.shift(1)) & (prices > prices.shift(-1))
        swing_lows = (prices < prices.shift(1)) & (prices < prices.shift(-1))
        
        return prices[swing_highs], prices[swing_lows]
    
    def _fibonacci_retracement(self, high: float, low: float, level: float) -> float:
        """Calculate Fibonacci retracement level."""
        return high - (high - low) * level
    
    def classify_pattern(self, wave_points):
        """
        Classify Elliott Wave pattern as impulsive or corrective
        
        Impulsive Wave Criteria:
        - Wave 2 retraces less than 61.8% of Wave 1
        - Wave 3 is typically the longest wave
        - Wave 4 does not overlap with Wave 1
        - Wave 5 follows specific length rules
        """
        if not wave_points:
            return "Insufficient Data"
        
        # Extract wave details using the new dictionary structure
        wave_1 = wave_points['Wave_1']['price_range']
        wave_2 = wave_points['Wave_2']['price_range']
        wave_3 = wave_points['Wave_3']['price_range']
        
        # Calculate wave lengths
        wave_1_length = abs(wave_1[1] - wave_1[0])
        wave_2_length = abs(wave_2[1] - wave_2[0])
        wave_3_length = abs(wave_3[1] - wave_3[0])
        
        # Retracement check
        wave_2_retracement = wave_2_length / wave_1_length
        
        # Preliminary impulsive wave checks
        is_impulsive = (
            0.382 <= wave_2_retracement <= 0.618 and  # Fibonacci retracement
            wave_3_length > wave_1_length  # Wave 3 typically longest
        )
        
        return "Impulsive" if is_impulsive else "Corrective"

    def _detect_swings(self, prices: pd.Series, window: int = 5) -> (pd.Series, pd.Series):
        """Detect swing highs and lows."""
        swing_highs = (prices > prices.shift(1)) & (prices > prices.shift(-1))
        swing_lows = (prices < prices.shift(1)) & (prices < prices.shift(-1))
        
        # Ensure chronological order and remove conflicting points
        swing_highs = prices[swing_highs]
        swing_lows = prices[swing_lows]
        
        # Sort both swing highs and lows by index to maintain chronological sequence
        swing_highs = swing_highs.sort_index()
        swing_lows = swing_lows.sort_index()
        
        return swing_highs, swing_lows

    def identify_wave(self):
        """
        Identify the Elliott Wave pattern with timeframe information.
        """
        try:
            prices = self.data['Close']
            swing_highs, swing_lows = self._detect_swings(prices)

            wave_points = []
            for i in range(len(swing_lows) - 1):
                for j in range(i + 1, len(swing_highs)):
                    wave_1_low = swing_lows.iloc[i]
                    wave_1_high = swing_highs.iloc[j-1]
                    wave_2_low = swing_lows.iloc[j]
                    
                    # Ensure chronological order of dates
                    if swing_lows.index[i] < swing_highs.index[j-1] < swing_lows.index[j]:
                        # Additional wave identification logic
                        wave_1_length = wave_1_high - wave_1_low
                        wave_2_retracement = abs((wave_2_low - wave_1_high) / wave_1_length)

                        # Check for next potential wave high
                        if j < len(swing_highs):
                            wave_3_high = swing_highs.iloc[j]
                            
                            if 0.5 <= wave_2_retracement <= 0.618:
                                wave_points.append({
                                    'Wave_1': {
                                        'price_range': (wave_1_low, wave_1_high),
                                        'start_date': swing_lows.index[i],
                                        'end_date': swing_highs.index[j-1],
                                        'duration': swing_highs.index[j-1] - swing_lows.index[i]
                                    },
                                    'Wave_2': {
                                        'price_range': (wave_1_high, wave_2_low),
                                        'start_date': swing_highs.index[j-1],
                                        'end_date': swing_lows.index[j],
                                        'duration': swing_lows.index[j] - swing_highs.index[j-1]
                                    },
                                    'Wave_3': {
                                        'price_range': (wave_2_low, wave_3_high),
                                        'start_date': swing_lows.index[j],
                                        'end_date': swing_highs.index[j],
                                        'duration': swing_highs.index[j] - swing_lows.index[j]
                                    }
                                })

            # Return and process waves as before
            if not wave_points:
                print("No valid Elliott Wave patterns found.")
                return None

            latest_wave = wave_points[-1]
            print("Latest Elliott Wave Identified:")
            
            # Enhanced print with timeframe details
            for wave_name, wave_data in latest_wave.items():
                print(f"\n{wave_name}:")
                print(f"  Price Range: ${wave_data['price_range'][0]:.2f} - ${wave_data['price_range'][1]:.2f}")
                print(f"  Start Date: {wave_data['start_date'].date()}")
                print(f"  End Date: {wave_data['end_date'].date()}")
                print(f"  Duration: {wave_data['duration']}")

            pattern_type = self.classify_pattern(latest_wave)
            print("\nPattern Classification:", pattern_type)
            
            return latest_wave

        except Exception as e:
            print(f"Detailed Error in Wave Identification:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
# Main Routine
if __name__ == "__main__":
    while True:
        try:
            symbol = input("\nEnter stock symbol (or 'q' to quit): ").upper()
            if symbol == 'Q':
                print("Exiting the program. Thank you!")
                break

            period = input("Enter historical data period (e.g., '1mo', '3mo', '1y'): ").lower()
            interval = input("Enter historical data interval (e.g., '1d', '1wk', '1mo'): ").lower()

            analyzer = ElliottWaveAnalyzer(symbol, period=period, interval=interval)
            wave = analyzer.identify_wave()

            if wave:
                print("\nIdentified Elliott Wave Details:")
                print(wave)
        except Exception as e:
            print(f"Error: {e}")
