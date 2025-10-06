import os
import re
import spacy
import glob
import PyPDF2
from tqdm import tqdm


class TranscriptCleaner:
    """
    A class to clean earnings call transcripts.

    This class handles:
    - Extracting transcript text from PDF files, starting from the first 'Operator' (ignoring preamble noise)
    - Removing speaker labels (e.g., lines with only names/titles)
    - Removing bracketed sections (e.g., "[Operator Instructions]")
    - Removing empty lines and extra spaces
    - Optionally anonymizing all person names (PERSON) using spaCy, with robust word-boundary handling
    - Fast extraction using PyPDF2 for performance
    """

    def __init__(self, raw_dir: str = "../data/transcripts/raw", clean_dir: str = "../data/transcripts/cleaned"):
        """
        Initialize the TranscriptCleaner.

        Args:
            raw_dir (str): Path to the folder containing raw transcript PDFs.
            clean_dir (str): Path to the folder where cleaned transcripts will be saved as .txt files.
        """
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        os.makedirs(clean_dir, exist_ok=True)
        self.nlp = spacy.load("en_core_web_trf")  # SpaCy model for NER (Named Entity Recognition)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Fast extraction using PyPDF2: reads page by page and starts collecting text from the first 'Operator' found.

        Args:
            pdf_path (str): Path to the PDF file to extract text from.

        Returns:
            str: The extracted raw text from the PDF, starting from 'Operator' if found.
        """
        text = ""
        found_operator = False
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                if not found_operator:
                    idx = page_text.find("Operator")
                    if idx != -1:
                        page_text = page_text[idx:]
                        found_operator = True
                if found_operator:
                    text += page_text + "\n"
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean the transcript text by removing speaker labels, bracketed sections, extra spaces, and empty lines.

        Args:
            text (str): The raw transcript text to clean.

        Returns:
            str: The cleaned transcript text.
        """
        # Remove lines that are likely speaker labels or titles
        text = re.sub(r"^([A-Z][a-zA-Z .,'-]+)$", "", text, flags=re.MULTILINE)
        # Remove bracketed sections
        text = re.sub(r"\[.*?\]", "", text)
        # Remove multiple spaces and empty lines
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        return text

    def anonymize_persons(self, text: str) -> str:
        """
        Anonymize person names in the transcript using spaCy NER.
        Only replaces full words, avoiding partial replacements (e.g. 'Tim' in 'Time').

        Args:
            text (str): The cleaned transcript text to anonymize.

        Returns:
            str: The anonymized transcript text with person names replaced by [PERSON].
        """
        doc = self.nlp(text)
        anonymized = text
        # Collect entities to replace (start, end, label)
        entities = [(ent.start_char, ent.end_char, ent.label_, ent.text) for ent in doc.ents if ent.label_ == "PERSON"]
        # Replace from end to start to avoid messing up indices
        for start, end, label, ent_text in sorted(entities, reverse=True):
            # Only replace if entity is a full word (check boundaries)
            if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
                anonymized = f"{anonymized[:start]}[PERSON]{anonymized[end:]}"
                # anonymized = anonymized[:start] + "[PERSON]" + anonymized[end:]
        return anonymized

    def process_file(self, pdf_path: str, out_path: str) -> None:
        """
        Process a single PDF file: extract, clean, anonymize, and save as .txt.

        Args:
            pdf_path (str): Path to the PDF file to process.
            out_path (str): Path to save the cleaned and anonymized .txt file.
        """
        text = self.extract_text_from_pdf(pdf_path)
        cleaned = self.clean_text(text)
        anonymized = self.anonymize_persons(cleaned)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(anonymized)

    def process_all_pdfs(self) -> None:
        """
        Process all PDF files in the source directory using process_file.

        Iterates through all PDF files in raw_dir, cleans and anonymizes each, and saves the result in clean_dir.
        Uses glob for file search and tqdm for progress bar.
        """
        pdf_files = glob.glob(os.path.join(self.raw_dir, "**", "*.pdf"), recursive=True)
        for pdf_path in tqdm(pdf_files, desc="Processing transcripts"):
            try:
                # Extract ticker and date from path
                rel_path = os.path.relpath(pdf_path, self.raw_dir)
                parts = rel_path.split(os.sep)
                ticker = parts[0]
                date = os.path.splitext(parts[-1])[0]
                out_dir = os.path.join(self.clean_dir, ticker)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{date}.txt")
                self.process_file(pdf_path, out_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    cleaner = TranscriptCleaner()
    cleaner.process_all_pdfs()
