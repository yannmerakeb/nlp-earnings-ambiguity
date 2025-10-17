import pandas as pd
import re
import textstat
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.transcript_preprocessor import *
from typing import Callable, Optional
# If you want to use FinBERT, you can add the import here (e.g. transformers, torch)

# Load mapping_words.json as a dictionary
json_path = os.path.join(os.path.dirname(__file__), "../data/mapping_words.json")
with open(json_path, "r", encoding="utf-8") as f:
    WORDS_MAPPING_DICT = json.load(f)


'''class FeatureExtractor:
    """
    Extract NLP features from earnings call transcripts.

    Features include:
    - Hedging word counts
    - Readability scores (Flesch-Kincaid, Gunning Fog)
    - Positive/negative tone ratio (lexicon-based)
    - TF-IDF features
    - Placeholder for FinBERT sentiment
    """

    def __init__(self):
        """
        Initialize the FeatureExtractor.
        """
        self.hedging_words = WORDS_MAPPING_DICT['hedging']['words']
        self.positive_words = WORDS_MAPPING_DICT['positive']['words']
        self.negative_words = WORDS_MAPPING_DICT['negative']['words']

    def count_hedging_words(self, text: str) -> int:
        """
        Count occurrences of hedging words in the text.

        Args:
            text (str): Cleaned transcript string.
        Returns:
            int: Number of hedging words found.
        """
        text_lower = text.lower()
        return sum(text_lower.count(word) for word in self.hedging_words)

    def compute_readability(self, text: str) -> dict:
        """
        Compute readability metrics of the text.

        Args:
            text (str): Cleaned transcript string.
        Returns:
            dict: Dictionary with keys 'flesch' and 'gunning_fog'.
        """
        return {
            "flesch": textstat.flesch_reading_ease(text),
            "gunning_fog": textstat.gunning_fog(text)
        }

    def compute_tone_ratio(self, text: str) -> float:
        """
        Compute a simple tone score: (positive - negative) / total words.
        Returns 0 if text is empty.

        Args:
            text (str): Cleaned transcript string.
        Returns:
            float: Tone ratio score.
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        total_words = len(words)
        if total_words == 0:
            return 0.0
        pos_count = sum(word in self.positive_words for word in words)
        neg_count = sum(word in self.negative_words for word in words)
        return (pos_count - neg_count) / total_words




    def count_words_from_list(self, text: str, word_list: list) -> int:
        """
        Count occurrences of words from a given list in the text.
        Args:
            text (str): Transcript text.
            word_list (list): List of words to count.
        Returns:
            int: Number of occurrences.
        """
        text_lower = text.lower()
        return sum(text_lower.count(word) for word in word_list)

    def ratio_words_from_list(self, text: str, word_list: list) -> float:
        """
        Ratio of words from a given list to total words in the text.
        Args:
            text (str): Transcript text.
            word_list (list): List of words to count.
        Returns:
            float: Ratio.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return 0.0
        count = sum(word in word_list for word in words)
        return count / total_words

    def polarity_score(self, text: str) -> float:
        """
        Polarity score: (positive - negative) / total words.
        Args:
            text (str): Transcript text.
        Returns:
            float: Polarity score.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return 0.0
        pos_count = sum(word in self.positive_words for word in words)
        neg_count = sum(word in self.negative_words for word in words)
        return (pos_count - neg_count) / total_words

    def avg_sentence_length(self, text: str) -> float:
        """
        Average sentence length in words.
        """
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        words = re.findall(r"\b\w+\b", text)
        if len(sentences) == 0:
            return 0.0
        return len(words) / len(sentences)

    def total_words(self, text: str) -> int:
        """
        Total number of words in the text.
        """
        return len(re.findall(r"\b\w+\b", text))

    def total_sentences(self, text: str) -> int:
        """
        Total number of sentences in the text.
        """
        from nltk.tokenize import sent_tokenize
        return len(sent_tokenize(text))

    def count_modal_verbs(self, text: str) -> int:
        """
        Count modal verbs (may, might, could, should, would, can, must, will, shall).
        """
        modal_verbs = ["may", "might", "could", "should", "would", "can", "must", "will", "shall"]
        return self.count_words_from_list(text, modal_verbs)

    def ratio_modal_verbs(self, text: str) -> float:
        """
        Ratio of modal verbs to total words.
        """
        modal_verbs = ["may", "might", "could", "should", "would", "can", "must", "will", "shall"]
        return self.ratio_words_from_list(text, modal_verbs)




    def transform(self, df: pd.DataFrame, text_col: str = "transcript") -> pd.DataFrame:
        """
        Apply feature extraction to a DataFrame of transcripts.

        Args:
            df (pd.DataFrame): DataFrame with a column containing transcripts.
            text_col (str): Name of the column with transcript text.
        Returns:
            pd.DataFrame: Original DataFrame with added feature columns.
        """
        df_features = df.copy()
        # Hedging
        df_features["hedging_count"] = df_features[text_col].apply(lambda x: self.count_words_from_list(x, self.hedging_words))
        df_features["hedging_ratio"] = df_features[text_col].apply(lambda x: self.ratio_words_from_list(x, self.hedging_words))
        # Positive/Negative
        df_features["positive_count"] = df_features[text_col].apply(lambda x: self.count_words_from_list(x, self.positive_words))
        df_features["positive_ratio"] = df_features[text_col].apply(lambda x: self.ratio_words_from_list(x, self.positive_words))
        df_features["negative_count"] = df_features[text_col].apply(lambda x: self.count_words_from_list(x, self.negative_words))
        df_features["negative_ratio"] = df_features[text_col].apply(lambda x: self.ratio_words_from_list(x, self.negative_words))
        # Polarity
        df_features["polarity_score"] = df_features[text_col].apply(self.polarity_score)
        # Readability
        readability = df_features[text_col].apply(self.compute_readability)
        df_features["flesch"] = readability.apply(lambda x: x["flesch"])
        df_features["gunning_fog"] = readability.apply(lambda x: x["gunning_fog"])
        # Structure
        df_features["avg_sentence_len"] = df_features[text_col].apply(self.avg_sentence_length)
        df_features["total_words"] = df_features[text_col].apply(self.total_words)
        df_features["total_sentences"] = df_features[text_col].apply(self.total_sentences)
        # Modal verbs
        df_features["modal_verbs_count"] = df_features[text_col].apply(self.count_modal_verbs)
        df_features["modal_verbs_ratio"] = df_features[text_col].apply(self.ratio_modal_verbs)
        # Placeholder for semantic/embedding features
        # df_features["semantic_dispersion"] = ...
        return df_features

    def compute_tfidf_features(
        self,
        df: pd.DataFrame,
        text_col: str = "transcript",
        max_features: int = 100,
        vectorizer: TfidfVectorizer | None = None,
    ) -> pd.DataFrame:
        """
        Compute TF-IDF features for the transcripts.
        - Si un vectorizer est fourni, il est utilisé tel quel.
        - Sinon, crée un vectorizer via make_tfidf_vectorizer avec max_features.
        - Gère les NaN et convertit en str pour éviter les erreurs.
        Args:
            df (pd.DataFrame): DataFrame with transcript column.
            text_col (str): Name of the column with transcript text.
            max_features (int): Number of TF-IDF features to keep (utilisé si vectorizer=None).
            vectorizer (TfidfVectorizer | None): Vectorizer optionnel (pré-fitté ou à fitter ici).
        Returns:
            pd.DataFrame: DataFrame with TF-IDF features appended.
        """
        vect = vectorizer or self.make_tfidf_vectorizer(max_features=max_features)
        texts = df[text_col].fillna("").astype(str).tolist()
        tfidf_matrix = vect.fit_transform(texts)
        feature_names = [f"tfidf_{w}" for w in vect.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        df_out = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
        return df_out

    def make_tfidf_vectorizer(
        self,
        max_features: int = 30000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int | float = 2,
        max_df: float = 0.9,
        stop_words: str | None = "english",
        lowercase: bool = True,
        dtype = np.float32,
    ) -> TfidfVectorizer:
        """
        Crée un TfidfVectorizer configuré pour les transcripts (réutilisable entre train/test).
        Ne l'entraîne pas: appelez fit/transform en dehors.
        """
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            dtype=dtype,
        )

    def add_finbert_sentiment(self, df: pd.DataFrame, text_col: str = "transcript") -> pd.DataFrame:
        """
        Placeholder for FinBERT sentiment extraction.
        Args:
            df (pd.DataFrame): DataFrame with transcript column.
            text_col (str): Name of the column with transcript text.
        Returns:
            pd.DataFrame: DataFrame with FinBERT sentiment columns appended.
        """
        # Example: df["finbert_sentiment"] = ...
        # You can implement FinBERT sentiment extraction here
        return df'''


class LinguisticFeatureExtractor:
    """
    Focused extractor for linguistic/lexical features (no TF-IDF).

    Provides a simple interface to append linguistically-motivated columns to a
    transcripts DataFrame.
    """
    def __init__(self):
        """
        Initialize the extractor by loading lexicons from mapping_words.json.
        """
        self.hedging_words = WORDS_MAPPING_DICT['hedging']['words']
        self.positive_words = WORDS_MAPPING_DICT['positive']['words']
        self.negative_words = WORDS_MAPPING_DICT['negative']['words']

    @staticmethod
    def _count_words_from_list(text: str, word_list: list) -> int:
        """
        Count occurrences of words from a given list in text (case-insensitive).

        Args:
            text: The input text.
            word_list: List of words to search for.
        Returns:
            int: Total number of occurrences found in text.
        """
        text_lower = text.lower()
        return sum(text_lower.count(word) for word in word_list)

    @staticmethod
    def _ratio_words_from_list(text: str, word_list: list) -> float:
        """
        Compute the ratio of tokens that belong to word_list over total tokens.

        Args:
            text: The input text.
            word_list: List of words of interest.
        Returns:
            float: Ratio in [0, 1]. Returns 0.0 if the text is empty.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return 0.0
        count = sum(word in word_list for word in words)
        return count / total_words

    @staticmethod
    def _polarity_score(text: str, positive_words: list, negative_words: list) -> float:
        """
        Compute a simple polarity score.

        Formula: (positive_count - negative_count) / total_words

        Args:
            text: The input text.
            positive_words: List of words considered positive.
            negative_words: List of words considered negative.
        Returns:
            float: Polarity score in [-1, 1] (0.0 for empty text).
        """
        words = re.findall(r"\b\w+\b", text.lower())
        total_words = len(words)
        if total_words == 0:
            return 0.0
        pos_count = sum(word in positive_words for word in words)
        neg_count = sum(word in negative_words for word in words)
        return (pos_count - neg_count) / total_words

    @staticmethod
    def _avg_sentence_length(text: str) -> float:
        """
        Compute average sentence length in number of tokens.

        Args:
            text: The input text.
        Returns:
            float: Average number of words per sentence (0.0 if no sentences).
        """
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        words = re.findall(r"\b\w+\b", text)
        if len(sentences) == 0:
            return 0.0
        return len(words) / len(sentences)

    @staticmethod
    def _total_words(text: str) -> int:
        """
        Count total number of word tokens in text.

        Args:
            text: The input text.
        Returns:
            int: Number of tokens.
        """
        return len(re.findall(r"\b\w+\b", text))

    @staticmethod
    def _total_sentences(text: str) -> int:
        """
        Count the number of sentences as segmented by NLTK.

        Args:
            text: The input text.
        Returns:
            int: Number of sentences.
        """
        from nltk.tokenize import sent_tokenize
        return len(sent_tokenize(text))

    @staticmethod
    def _compute_readability(text: str) -> dict:
        """
        Compute basic readability metrics for a given text.

        Args:
            text: The input text.
        Returns:
            dict: A dictionary with keys:
                - 'flesch' (float): Flesch Reading Ease score.
                - 'gunning_fog' (float): Gunning Fog index.
        """
        return {
            "flesch": textstat.flesch_reading_ease(text),
            "gunning_fog": textstat.gunning_fog(text),
        }

    def transform(self, df: pd.DataFrame, text_col: str = "transcript") -> pd.DataFrame:
        """
        Append linguistic features to the input DataFrame.

        Args:
            df: Input DataFrame containing a column with transcripts.
            text_col: Name of the column that holds the transcript text.
        Returns:
            pd.DataFrame: A copy of df with additional linguistic feature columns:
                - hedging_count, hedging_ratio
                - positive_count, positive_ratio
                - negative_count, negative_ratio
                - polarity_score
                - flesch, gunning_fog
                - avg_sentence_len, total_words, total_sentences
                - modal_verbs_count, modal_verbs_ratio
        """
        df_features = df.copy()
        # Hedging
        df_features["hedging_count"] = df_features[text_col].apply(lambda x: self._count_words_from_list(x, self.hedging_words))
        df_features["hedging_ratio"] = df_features[text_col].apply(lambda x: self._ratio_words_from_list(x, self.hedging_words))
        # Positive/Negative
        df_features["positive_count"] = df_features[text_col].apply(lambda x: self._count_words_from_list(x, self.positive_words))
        df_features["positive_ratio"] = df_features[text_col].apply(lambda x: self._ratio_words_from_list(x, self.positive_words))
        df_features["negative_count"] = df_features[text_col].apply(lambda x: self._count_words_from_list(x, self.negative_words))
        df_features["negative_ratio"] = df_features[text_col].apply(lambda x: self._ratio_words_from_list(x, self.negative_words))
        # Polarity
        df_features["polarity_score"] = df_features[text_col].apply(lambda t: self._polarity_score(t, self.positive_words, self.negative_words))
        # Readability
        readability = df_features[text_col].apply(self._compute_readability)
        df_features["flesch"] = readability.apply(lambda x: x["flesch"])
        df_features["gunning_fog"] = readability.apply(lambda x: x["gunning_fog"])
        # Structure
        df_features["avg_sentence_len"] = df_features[text_col].apply(self._avg_sentence_length)
        df_features["total_words"] = df_features[text_col].apply(self._total_words)
        df_features["total_sentences"] = df_features[text_col].apply(self._total_sentences)
        # Modal verbs
        modal_verbs = ["may", "might", "could", "should", "would", "can", "must", "will", "shall"]
        df_features["modal_verbs_count"] = df_features[text_col].apply(lambda x: self._count_words_from_list(x, modal_verbs))
        df_features["modal_verbs_ratio"] = df_features[text_col].apply(lambda x: self._ratio_words_from_list(x, modal_verbs))
        return df_features


class TfidfFeatureExtractor:
    """
    Thin wrapper around TfidfVectorizer with fit/transform helpers for DataFrames.

    Typical usage:
        tfidf = TfidfFeatureExtractor(max_features=20000)
        tfidf.fit(train_df, text_col="transcript")
        df_train = tfidf.transform(train_df)
        df_test = tfidf.transform(test_df)
    """
    def __init__(
        self,
        max_features: int = 30000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: float = 0.05,
        max_df: float = 0.8,
        stop_words: str | None = "english",
        lowercase: bool = False,
        dtype = np.float32,
    ) -> None:
        """
        Configure the underlying TfidfVectorizer with sensible defaults.

        Args:
            max_features: Cap on the vocabulary size.
            ngram_range: N-gram range to consider (e.g. (1, 2) for unigrams+bigrams).
            min_df: Ignore terms that appear in fewer than min_df documents.
            max_df: Ignore terms that appear in more than max_df fraction of docs.
            stop_words: Stopwords to remove (e.g. 'english').
            lowercase: Whether to lowercase the text.
            dtype: Numeric dtype for the TF-IDF matrix.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            lowercase=lowercase,
            token_pattern=r"(?u)\b[\w\[\]]+\b",
            dtype=dtype,
        )
        self.text_col = "transcript_tfidf"

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the TF-IDF vocabulary and append TF-IDF features to the DataFrame.
        """
        texts = df[self.text_col].tolist()
        mat = self.vectorizer.fit_transform(texts)
        feature_names = [f"tfidf_{w}" for w in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(mat.toarray(), columns=feature_names)
        final_df = (pd.concat([df[['ticker', 'date']], tfidf_df], axis=1)
                    .sort_values(by='date', ascending=True).reset_index(drop=True))

        return final_df



"""if __name__ == "__main__":

    # Load preprocessed transcripts and extract features
    transcript_preprocessor = TranscriptPreprocessor()
    df = transcript_preprocessor.load_transcripts()

    tfidf_preprocessor = TfidfFeatureExtractor()
    df1 = tfidf_preprocessor.fit_transform(df)
    # df = tfidf_preprocessor.tfidf_from_preprocessed_dir()"""



