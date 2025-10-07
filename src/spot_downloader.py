import pandas as pd
import logging
import pandas_datareader.data as web
import datetime as dt
import os
from transcript_loader import TranscriptLoader

class SpotDownloader:
    """
    Download daily close prices for a list of tickers using pandas_datareader (stooq) and save all to a single CSV file.
    The file will be named 'ALLTICKERS_startdate_enddate.csv' and stored in data/spot/.
    """
    def __init__(self, tickers: list, start_date: dt, end_date: dt, output_dir: str = "../data/spot"):
        # Initialize SpotDownloader with tickers, date range, and output directory
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def download_single(self, ticker: str):
        """
        Download daily close prices for a single ticker from stooq.
        Returns a DataFrame or None if failed.
        """
        try:
            # Download price data for the given ticker
            print(f"Downloading {ticker}...")
            df = web.DataReader(ticker, 'stooq', self.start_date, self.end_date)

            if df.empty:
                # No data found for ticker
                print(f"No data for {ticker}")
                return None

            # Format the DataFrame
            df = df.reset_index()
            df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
            df["ticker"] = ticker
            print(f"Prices for {ticker} downloaded.")
            return df

        except Exception as e:
            # Handle download error
            print(f"Error downloading {ticker}: {e}")
            return None

    def download_all(self):
        """
        Download daily close prices for all tickers one by one.
        Save all results in a single CSV file. Log failed tickers.
        """
        # Start downloading for all tickers
        print(f"Downloading prices for tickers: {self.tickers}\nFrom {self.start_date} to {self.end_date}")
        all_prices = []
        failed_tickers = []

        # Download prices for each ticker
        for ticker in self.tickers:
            df = self.download_single(ticker)
            if df is not None:
                all_prices.append(df)
            else:
                failed_tickers.append(ticker)

        if all_prices:
            # Concatenate all DataFrames and save to CSV
            prices_df = pd.concat(all_prices, ignore_index=True)
            prices_df = prices_df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
            filename = f"{self.start_date.strftime('%Y-%m-%d')}_{self.end_date.strftime('%Y-%m-%d')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            prices_df.to_csv(filepath, index=False)
            print(f"Prices for all tickers saved to {filepath}")
        else:
            # No prices were downloaded
            print("No prices downloaded.")
        if failed_tickers:
            # Log tickers that failed to download
            print(f"Tickers failed: {failed_tickers}")

if __name__ == "__main__":
    # Load tickers from transcripts
    loader = TranscriptLoader()
    df_transcripts = loader.load_all()
    tickers = df_transcripts['ticker'].unique().tolist()

    # Define date range
    start_date = df_transcripts['date'].min() - dt.timedelta(days=30)
    end_date = df_transcripts['date'].max() + dt.timedelta(days=30)

    # Download spot prices for all tickers one by one
    downloader = SpotDownloader(tickers, start_date, end_date)
    downloader.download_all()
