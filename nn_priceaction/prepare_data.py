import pandas as pd

def fix_data(csv_path: str, fixed_csv_name: str) -> None:

    # Read the CSV, set the Date column as the index
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Reset the index so the Date column is a regular column again
    df.reset_index(inplace=True)

    # Now apply pd.to_datetime() to the Date column
    df["Date"] = pd.to_datetime(df["Date"], format='%d.%m.%Y %H:%M:%S.%f')

    # Set the Date column back as the index if needed
    df.set_index("Date", inplace=True)

    # Sort the DataFrame by the Date index
    df.sort_index(inplace=True)

    # Drop NaN values
    df.dropna(inplace=True)

    # Save the cleaned DataFrame
    df.to_csv(f'./{fixed_csv_name}.csv')

def load_and_transform(file_path: str):
        # Load CSV with proper datetime handling
        df = pd.read_csv(
            file_path,
            delimiter='\t',
            parse_dates={'datetime': ['<DATE>', '<TIME>']},
            date_parser=lambda x: pd.to_datetime(x, format='%Y.%m.%d %H:%M:%S'),
            usecols=['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<SPREAD>']
        )
        
        # Clean column names
        df.columns = df.columns.str.strip('<>').str.lower()
        
        # Column renaming and reorganization
        df = df.rename(columns={'tickvol': 'vol'})
        df = df[['datetime', 'open', 'high', 'low', 'close', 'vol', 'spread']]
        
        return df.sort_values('datetime').reset_index(drop=True)

def calculate_condition_met(df, stop_distance, rr_ratio, lookahead):
    # Initialize both condition columns
    
    # 0 -> NO TRADE
    # 1 -> LONG
    # 2 -> SHORT

    df['order_type'] = 0
    
    for i in range(len(df)):
        current = df.iloc[i]
        future = df.iloc[i+1:i+lookahead+1]
        
        if future.empty:
            break
        
        # Calculate entry prices with spread adjustment
        long_entry = current['close'] + current['spread'] * 0.01
        short_entry = current['close'] - current['spread'] * 0.01
        
        # Calculate price levels with spread
        long_stop = long_entry - stop_distance
        long_target = long_entry + (stop_distance * rr_ratio)
        short_stop = short_entry + stop_distance
        short_target = short_entry - (stop_distance * rr_ratio)
        
        # Track both scenarios
        long_status = {'met': False, 'stopped': False}
        short_status = {'met': False, 'stopped': False}
        
        for _, future_candle in future.iterrows():
        # Check long condition if not yet resolved
            if not long_status['met'] and not long_status['stopped']:
                # Check if price hit stop loss first
                if future_candle['low'] <= long_stop:
                    long_status['stopped'] = True
                # Check if price hit take profit first
                elif future_candle['high'] >= long_target:
                    long_status['met'] = True

            # Check short condition if not yet resolved
            if not short_status['met'] and not short_status['stopped']:
                # Check if price hit stop loss first
                if future_candle['high'] >= short_stop:
                    short_status['stopped'] = True
                # Check if price hit take profit first
                elif future_candle['low'] <= short_target:
                    short_status['met'] = True

            # Early exit if both directions are resolved
            # if (long_status['met'] or long_status['stopped']) and \
            #     (short_status['met'] or short_status['stopped']):
            #     break

        # Record results in DataFrame
        order_type = 0
        if long_status['met']: order_type = 1
        elif short_status['met']: order_type = 2
        
        df.at[i, 'order_type'] = order_type

    return df

if __name__ == '__main__':

    csv_path = 'Data/MT5/XAUUSD_H1_201708100000_202502282300.csv'