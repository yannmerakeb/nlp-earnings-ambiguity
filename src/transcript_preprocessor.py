import os
import re
import glob
from typing import List
import spacy
import PyPDF2
import nltk
from tqdm import tqdm

class TranscriptPreprocessor:
    """
    Class for transcript preprocessing.

    Handles:
    - Extracting transcript text from PDF files (starting from 'Operator')
    - Cleaning text (removing speaker labels, bracketed sections, extra spaces)
    - Anonymizing person names using spaCy NER
    - Normalizing text (lowercase, remove anonymization tags, non-ASCII chars)
    - Segmenting text into sentences (NLTK)
    - Saving processed files

    Designed for readability scoring and NLP feature extraction.
    """

    def __init__(self, raw_dir: str = "../data/transcripts/raw", preprocessed_dir: str = "../data/transcripts/preprocessed"):
        """
        Initialize the TranscriptPreprocessor.

        Args:
            raw_dir (str): Path to raw transcript PDFs.
            preprocessed_dir (str): Path to save fully preprocessed transcripts.
        """
        self.raw_dir = raw_dir
        self.preprocessed_dir = preprocessed_dir
        os.makedirs(preprocessed_dir, exist_ok=True)
        self.nlp = spacy.load("en_core_web_trf")
        nltk.download('punkt', quiet=True)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract transcript text from PDF, starting from first 'Operator'.

        Args:
            pdf_path (str): Path to PDF file.

        Returns:
            str: Extracted raw text.
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
        Clean transcript text by removing speaker labels, bracketed sections, extra spaces, and empty lines.

        Args:
            text (str): Raw transcript text.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r"^([A-Z][a-zA-Z .,'-]+)$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        return text

    def anonymize_persons(self, text: str) -> str:
        """
        Anonymize person names using spaCy NER.

        Args:
            text (str): Cleaned transcript text.

        Returns:
            str: Text with person names replaced by [PERSON].
        """
        doc = self.nlp(text)
        anonymized = text
        entities = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
        for start, end in sorted(entities, reverse=True):
            if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
                anonymized = f"{anonymized[:start]}[PERSON]{anonymized[end:]}"
        return anonymized

    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove extra spaces, anonymization tags, and non-ASCII chars.

        Args:
            text (str): Anonymized transcript text.

        Returns:
            str: Normalized text.
        """
        text = text.lower()
        text = re.sub(r"\[person\]", "", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences using NLTK.

        Args:
            text (str): Normalized transcript text.

        Returns:
            List[str]: List of sentences.
        """
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def process_pdf(self, pdf_path: str, preprocessed_path: str) -> None:
        """
        Full pipeline for a single PDF: extract, clean, anonymize, normalize, segment, and save.

        Args:
            pdf_path (str): Path to PDF file.
            preprocessed_path (str): Path to save fully preprocessed text.
        """
        text = self.extract_text_from_pdf(pdf_path)
        cleaned = self.clean_text(text)
        anonymized = self.anonymize_persons(cleaned)
        normalized = self.normalize_text(anonymized)
        sentences = self.segment_sentences(normalized)
        preprocessed = " ".join(sentences)
        with open(preprocessed_path, "w", encoding="utf-8") as f:
            f.write(preprocessed)

    def process_all_pdfs(self) -> None:
        """
        Process all PDF files in raw_dir and save preprocessed versions.

        Uses glob for file search and tqdm for progress bar.
        """
        pdf_files = glob.glob(os.path.join(self.raw_dir, "**", "*.pdf"), recursive=True)
        for pdf_path in tqdm(pdf_files, desc="Processing transcripts"):
            try:
                rel_path = os.path.relpath(pdf_path, self.raw_dir)
                parts = rel_path.split(os.sep)
                ticker = parts[0]
                date = os.path.splitext(parts[-1])[0]
                preprocessed_dir = os.path.join(self.preprocessed_dir, ticker)
                os.makedirs(preprocessed_dir, exist_ok=True)
                preprocessed_path = os.path.join(preprocessed_dir, f"{date}.txt")
                self.process_pdf(pdf_path, preprocessed_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    preprocessor = TranscriptPreprocessor()
    preprocessor.process_all_pdfs()
