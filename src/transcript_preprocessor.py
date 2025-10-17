import pandas as pd
import re
import glob
import spacy
import PyPDF2
import nltk
from tqdm import tqdm
from typing import Optional
from spacy.language import Language
import os
from datetime import datetime


class TranscriptPreprocessor:
    """
    Transcript preprocessing pipeline:
    1. Extract raw text from PDF
    2. Normalize speaker boundaries
    3. Remove all 'Operator' sections
    4. Keep only paragraphs of the two main speakers
    5. Clean text (remove labels, brackets)
    6. Normalize spaces
    7. Anonymize entities (spaCy NER)
    8. Apply TF-IDF preprocessing (remove numbers, special chars, lemmatize)
    9. Apply BERT preprocessing (optional lowercasing, preserve structure)
    10. Save processed text:
        # - Anonymized: preprocessed/<ticker>/<date>.txt
        - TF-IDF: preprocessed/tfidf/<ticker>/<date>.txt
        - BERT: preprocessed/bert/<ticker>/<date>.txt
    """

    def __init__(self, raw_dir: str = "../data/transcripts/raw",
                 preprocessed_dir: str = "../data/transcripts/preprocessed"):
        self.raw_dir = raw_dir
        self.preprocessed_dir = preprocessed_dir
        os.makedirs(preprocessed_dir, exist_ok=True)
        # Create tfidf and bert subfolders
        os.makedirs(os.path.join(preprocessed_dir, "tfidf"), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_dir, "bert"), exist_ok=True)
        # Load spaCy model once, used for both NER and lemmatization
        self.nlp: Language = spacy.load("en_core_web_trf")
        self.REPLACE_LABELS = {"PRODUCT", "ORG", "PERSON", "DATE", "TIME"}
        self.DROP_LABELS = {"LAW", "WORK_OF_ART", "EVENT", "LANGUAGE"}
        nltk.download('punkt', quiet=True)
        # Precompile regex for speaker-only lines
        self._speaker_line_re = re.compile(
            r"(?m)^("
            r"Operator"
            r"|"
            r"[A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+){1,5}"
            r"|"
            r"[A-Z]{2,}(?: [A-Z]{2,}){1,5}"
            r"):?\s*$"
        )

    # --- Utility Methods ---
    def _split_by_speakers(self, text: str):
        """
        Split text into (speaker, content) segments based on speaker-only lines.
        """
        parts = self._speaker_line_re.split(text)
        segments = []
        for i in range(1, len(parts), 2):
            speaker = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append((speaker, content))
        return segments

    def _collect_known_speaker_names(self, text: str) -> set[str]:
        """
        Collect Title Case speaker names, excluding 'Operator'.
        """
        names = set()
        for speaker, _ in self._split_by_speakers(text):
            sp = speaker.strip()
            if not sp or sp.lower() == "operator":
                continue
            if re.fullmatch(r"[A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+){1,7}", sp):
                names.add(sp)
        return names

    @staticmethod
    def _collapse_name(s: str) -> str:
        """
        Collapse a name by removing non-letters and lowercasing.
        """
        return re.sub(r"[^A-Za-z]", "", s).lower()

    # --- Preprocessing Steps ---
    '''def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract transcript text from PDF, starting from the first occurrence of 'Operator'.
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
                    idx = page_text.lower().find("operator")
                    if idx != -1:
                        page_text = page_text[idx:]
                        found_operator = True
                if found_operator:
                    text += page_text + "\n"
        return text'''

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract transcript text from PDF, starting from the third newline.
        """
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                text += page_text + "\n"

        # Find the third newline and return text from that point
        newline_count = 0
        for i, char in enumerate(text):
            if char == '\n':
                newline_count += 1
                if newline_count == 3:
                    return text[i + 1:]  # Start after the third newline

        # If less than 3 newlines found, return the whole text
        return text


    def normalize_speaker_boundaries(self, text: str) -> str:
        """
        Normalize speaker lines for merged-name edge cases.
        """
        known = self._collect_known_speaker_names(text)
        if not known:
            return text
        mapping = {self._collapse_name(n): n for n in known}
        for collapsed, proper in mapping.items():
            pattern_with_content = re.compile(rf"(?im)^\s*({collapsed})(:?)\s+(?=\S)")
            text = pattern_with_content.sub(lambda m: f"{proper}\n", text)
            pattern_line_only = re.compile(rf"(?im)^\s*({collapsed})(:?)\s*$")
            text = pattern_line_only.sub(lambda m: f"{proper}", text)
        return text

    def drop_operator_sections(self, text: str) -> str:
        """
        Remove all 'Operator' lines and their following paragraphs.
        """
        segments = self._split_by_speakers(text)
        kept_chunks = []
        for speaker, content in segments:
            if speaker.lower() == "operator":
                continue
            kept_chunks.append(f"{speaker}\n{content.strip()}".strip())
        return "\n".join(chunk for chunk in kept_chunks if chunk)

    def extract_top_speakers_text(self, text: str) -> str:
        """
        Keep only paragraphs of the two speakers with the largest content (by word count).
        """
        segments = self._split_by_speakers(text)
        word_counts = {}
        for speaker, content in segments:
            if speaker.lower() == "operator":
                continue
            wc = len(re.findall(r"\b\w+\b", content))
            word_counts[speaker] = word_counts.get(speaker, 0) + wc
        top_two = set(sorted(word_counts, key=word_counts.get, reverse=True)[:2])
        kept_chunks = []
        for speaker, content in segments:
            if speaker in top_two:
                kept_chunks.append(f"{speaker}\n{content.strip()}".strip())
        return "\n".join(kept_chunks)

    def clean_text(self, text: str) -> str:
        """
        Clean transcript text:
        - Remove [ ... ] bracketed snippets (single-line)
        - Drop speaker-only label lines
        """
        text = re.sub(r"\[[^\]]{0,50}\]", "", text, flags=re.DOTALL)
        text = re.sub(r"(?m)^[A-Z][A-Za-z'.-]+(?: [A-Z][A-Za-z'.-]+){1,5}\s*$", "", text)
        return text

    def normalize_spaces(self, text: str) -> str:
        """
        Normalize spaces: replace multiple spaces, tabs, newlines with a single space.
        """
        return re.sub(r"\s+", " ", text).strip()

    def anonymize_entities(self, text: str) -> str:
        """
        Anonymize named entities in REPLACE_LABELS with their LABEL. Drop entities in DROP_LABELS.
        """
        # Process text with spaCy NLP pipeline
        doc = self.nlp(text)
        original = text
        out = text

        # Collect entity spans (start, end, label) for replacement or dropping
        spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
                 if ent.label_ in self.REPLACE_LABELS or ent.label_ in self.DROP_LABELS]

        # Process spans in reverse order to avoid index shifting
        for start, end, label in sorted(spans, reverse=True):
            # Ensure entity is not part of a larger word (check boundaries)
            if (start == 0 or not original[start - 1].isalnum()) and (
                    end == len(original) or not original[end].isalnum()):
                if label in self.REPLACE_LABELS:
                    # Replace entity with label
                    out = f"{out[:start]}{label}{out[end:]}"
                elif label in self.DROP_LABELS:
                    # Drop entity entirely
                    out = f"{out[:start]}{out[end:]}"

        return out


    def tfidf_preprocess(self, text: str) -> str:
        """
        Preprocessing for TF-IDF:
        - Remove pure numbers (e.g., "58.3", "2023", but keep "Q3")
        - Remove special characters (except brackets for [PERSON], [ORG])
        - Lemmatize
        - Normalize spaces
        """
        # Remove pure numbers (e.g., "58.3", "2023", but keep "Q3")
        text = re.sub(r"\b\d+(?:[\d,._-]*\d)?\b", " ", text)

        # Remove alphanumeric combinations (Q3, 50th, COVID19, etc.)
        text = re.sub(r'\b[a-zA-Z]*\d+[a-zA-Z]*\b', ' ', text)

        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Lemmatization using self.nlp
        doc = self.nlp(text, disable=["ner", "parser"])
        tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        # Join and normalize spaces
        text = " ".join(tokens)
        text = self.normalize_spaces(text)
        return text

    def bert_preprocess(self, text: str, lowercase: bool = True) -> str:
        """
        Preprocessing for BERT:
        - Optionally lowercase (for uncased models like bert-base-uncased)
        - Normalize spaces
        """
        # Optional lowercasing
        text = text.lower() if lowercase else text
        # Normalize spaces (already minimal due to prior steps)
        text = self.normalize_spaces(text)
        return text

    # --- Main Pipeline Methods ---
    def process_pdf(self, pdf_path: str) -> None:
        """
        Full pipeline for a single PDF: extract, normalize speaker boundaries, drop 'Operator',
        keep top speakers, clean, normalize spaces, anonymize, apply TF-IDF and BERT preprocessing.
        Returns (anonymized_text, tfidf_text, bert_text).
        """
        # Get the ticker and date from the pdf_path
        rel = os.path.relpath(pdf_path, self.raw_dir)
        parts = rel.split(os.sep)
        ticker = parts[0]
        date = os.path.splitext(parts[-1])[0]

        # Output paths by ticker
        tfidf_dir = os.path.join(self.preprocessed_dir, "tfidf", ticker)
        bert_dir = os.path.join(self.preprocessed_dir, "bert", ticker)
        os.makedirs(tfidf_dir, exist_ok=True)
        os.makedirs(bert_dir, exist_ok=True)
        tfidf_exists = False
        bert_exists = False

        tfidf_fp = os.path.join(tfidf_dir, f"{date}.txt")
        bert_fp = os.path.join(bert_dir, f"{date}.txt")

        # Write only if the file does not already exist
        if os.path.exists(tfidf_fp):
            tfidf_exists = True
            print(f"TF-IDF file exists, skipping: {tfidf_fp}")

        if os.path.exists(bert_fp):
            bert_exists = True
            print(f"BERT file exists, skipping: {bert_fp}")

        if tfidf_exists and bert_exists:
            return

        # Pipeline steps
        raw_text = self.extract_text_from_pdf(pdf_path)
        normalized_boundaries_text = self.normalize_speaker_boundaries(raw_text)
        no_operator_text = self.drop_operator_sections(normalized_boundaries_text)
        top_speakers_text = self.extract_top_speakers_text(no_operator_text)
        cleaned_text = self.clean_text(top_speakers_text)
        normalized_text = self.normalize_spaces(cleaned_text)
        anonymized_text = self.anonymize_entities(normalized_text)

        if not tfidf_exists:
            tfidf_text = self.tfidf_preprocess(anonymized_text)
            with open(tfidf_fp, "w", encoding="utf-8") as f:
                f.write(tfidf_text)

        """if not bert_exists:
            bert_text = self.bert_preprocess(anonymized_text, lowercase=True)  # Default to lowercase for BERT
            with open(bert_fp, "w", encoding="utf-8") as f:
                f.write(bert_text)"""


    def process_all_pdfs(self): # -> pd.DataFrame:
        """
        Process all PDFs under raw_dir, apply full preprocessing pipeline including TF-IDF and BERT,
        and return a DataFrame with anonymized, TF-IDF, and BERT processed texts.
        """
        pdf_files = glob.glob(os.path.join(self.raw_dir, "**", "*.pdf"), recursive=True)
        for pdf_path in tqdm(pdf_files, desc="Processing transcripts"):
            try:
                self.process_pdf(pdf_path)
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    def load_transcripts(self, pattern: str = "**/*.txt") -> pd.DataFrame:
        """
        Load preprocessed .txt files into a DataFrame, combining TF-IDF and BERT
        transcripts for each (ticker, date) pair into a single row.
        """
        root = self.preprocessed_dir
        files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))

        # Group data by (ticker, date)
        data = {}
        for fp in files:
            # Read file content
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()

            # Extract subfolder (tfidf/bert), ticker, and date from file path
            rel = os.path.relpath(fp, root).split(os.sep)
            subfolder, ticker, date = rel[0], rel[1], os.path.splitext(rel[-1])[0]

            # Use (ticker, date) as key
            date = datetime.strptime(date, "%Y%m%d")
            key = (ticker, date)
            if key not in data:
                data[key] = {"ticker": ticker, "date": date}

            # Assign text to tfidf or bert column
            data[key]["transcript_" + subfolder] = text

        # Convert to DataFrame and ensure column order
        df = pd.DataFrame(list(data.values()))[["ticker", "date", "transcript_tfidf", "transcript_bert"]].sort_values('date', ascending=True).reset_index(drop=True)

        # Pivot to have tickers as columns
        # df = df.pivot(index='date', columns='ticker', values='transcript_tfidf')

        return df


'''if __name__ == "__main__":
    a = TranscriptPreprocessor()
    a.load_transcripts()'''