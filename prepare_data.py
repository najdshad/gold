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


if __name__ == '__main__':
    ...