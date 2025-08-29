import os
import re
import spacy


class TranscriptCleaner:
    """
    A class to clean earnings call transcripts.

    This class handles:
    - Removing speaker labels (e.g., "Name -- Title")
    - Removing bracketed sections (e.g., "[Operator Instructions]")
    - Removing empty lines and extra spaces
    - Optionally anonymizing all proper names (PERSON) using spaCy
    """

    def __init__(self, raw_dir: str = "../data/transcripts/raw", clean_dir: str = "../data/transcripts/cleaned"):
        """
        Initialize the cleaner with input and output directories.

        Args:
        - raw_dir: Path to folder containing raw transcripts.
        - clean_dir: Path to folder where cleaned transcripts will be saved.
        """
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        os.makedirs(clean_dir, exist_ok=True)
        self.nlp = spacy.load("en_core_web_sm")  # SpaCy model for NER (Name Entity Recognition)

    def clean_text(self, text: str, anonymize_names: bool = False) -> str:
        """
        Clean a single transcript string.

        Args:
        - text: Raw transcript text.
        - anonymize_names: If True, replace all proper names with a generic token.

        Returns:
        - Cleaned transcript text.
        """
        # Remove speaker labels "Name -- Title"
        text = re.sub(r'^[A-Za-z\s]+ -- [^\n]+', '', text, flags=re.MULTILINE)

        # Remove bracketed instructions like [Operator Instructions], [Music], etc.
        text = re.sub(r'\[.*?\]', '', text)

        # Remove standalone mentions of 'Operator'
        text = re.sub(r'^\s*Operator\s*$', '', text, flags=re.MULTILINE)

        # Optionally anonymize proper names
        if anonymize_names:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    text = text.replace(ent.text, "SPEAKER")

        # Remove empty lines and extra whitespace
        text = re.sub(r'\n+', '\n', text).strip()

        return text

    def process_file(self, filepath: str, save_path: str, anonymize_names: bool = False) -> None:
        """
        Read a raw transcript file, clean it, and save it.

        Args:
        - filepath: Path to raw transcript file.
        - save_path: Path to save the cleaned transcript.
        - anonymize_names: If True, replace all proper names with a generic token.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        cleaned_text = self.clean_text(text, anonymize_names)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

    def process_all(self, anonymize_names: bool = False) -> None:
        """
        Process all transcript files in raw_dir and save cleaned files in clean_dir.

        The directory structure is preserved: raw_dir/TICKER/YYYYMMDD.txt
        → clean_dir/TICKER/YYYYMMDD.txt

        Args:
        - anonymize_names: If True, replace all proper names with a generic token.
        """
        for ticker in os.listdir(self.raw_dir):
            ticker_raw_path = os.path.join(self.raw_dir, ticker)
            ticker_clean_path = os.path.join(self.clean_dir, ticker)
            os.makedirs(ticker_clean_path, exist_ok=True)

            for filename in os.listdir(ticker_raw_path):
                if filename.endswith('.txt'):
                    raw_file = os.path.join(ticker_raw_path, filename)
                    clean_file = os.path.join(ticker_clean_path, filename)
                    self.process_file(raw_file, clean_file, anonymize_names)
                    print(f"Cleaned {raw_file} → {clean_file}")


if __name__ == "__main__":
    cleaner = TranscriptCleaner()
    cleaner.process_all(anonymize_names=True)  # Set True to anonymize all names
