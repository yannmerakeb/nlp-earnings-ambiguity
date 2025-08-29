import os
import pandas as pd

class TranscriptLoader:
    """
    Load all cleaned transcripts into a pandas DataFrame.

    The DataFrame has columns: ['ticker', 'date', 'transcript']
    """

    def __init__(self, cleaned_dir: str = "../data/transcripts/cleaned"):
        """
        Args:
        - cleaned_dir: Path to folder containing cleaned transcripts
        """
        self.cleaned_dir = cleaned_dir

    def load_all(self) -> pd.DataFrame:
        """
        Load all transcripts and return a DataFrame.

        Returns:
        - df: pandas DataFrame with columns ['ticker', 'date', 'transcript']
        """
        records = []

        for ticker in os.listdir(self.cleaned_dir):
            ticker_path = os.path.join(self.cleaned_dir, ticker)
            if not os.path.isdir(ticker_path):
                continue

            for filename in os.listdir(ticker_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(ticker_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        transcript = f.read()

                    # Extract date from filename
                    date_str = os.path.splitext(filename)[0]
                    records.append({
                        "ticker": ticker,
                        "date": date_str,
                        "transcript": transcript
                    })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        return df


if __name__ == "__main__":
    loader = TranscriptLoader()
    df = loader.load_all()
    print(df.head())