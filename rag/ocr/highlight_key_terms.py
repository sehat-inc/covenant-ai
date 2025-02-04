"""
Author: Hamza Amin
Dated: 1st Feb 2025
Purpose: Module to highlight key terms in a PDF document based on a summary.
         This module is intended for integration with a Flask application.
"""

import io
import fitz
import time
import re
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Union, Optional

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration settings for the highlighting system."""
    model_name: str = 'all-MiniLM-L6-v2'
    similarity_threshold: float = 0.63
    min_sentence_length: int = 10


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class PDFExtractionError(DocumentProcessingError):
    """Raised when PDF text extraction fails."""
    pass


class ModelLoadingError(DocumentProcessingError):
    """Raised when the transformer model fails to load."""
    pass


class TextProcessor:
    """Handles text processing operations."""

    def __init__(self):
        try:
            self.stopwords: Set[str] = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Failed to load NLTK stopwords: {str(e)}")
            raise DocumentProcessingError(f"Failed to load NLTK stopwords: {str(e)}")
        self.legal_important_words: Set[str] = {
            'shall', 'must', 'will', 'not', 'no', 'nor', 'any', 'all', 'none',
            'may', 'might', 'can', 'cannot', 'should', 'would', 'hereby'
        }
        # Preserve legal terms in the text by removing them from the stopwords list.
        self.stopwords -= self.legal_important_words

    def preprocess_text(self, text: str) -> str:
        """Preprocess text while preserving legal terms."""
        try:
            text = text.lower()
            # Retain word characters, whitespace, and periods.
            text = re.sub(r'[^\w\s.]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            words = text.split()
            # Remove stopwords unless they are legal important words.
            words = [word for word in words if (word not in self.stopwords or word in self.legal_important_words)]
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            raise DocumentProcessingError(f"Text preprocessing failed: {str(e)}")


class DocumentProcessor:
    """Handles document processing operations."""

    def __init__(self, config: Config):
        self.config = config
        self.text_processor = TextProcessor()
        try:
            self.model = SentenceTransformer(config.model_name)
        except Exception as e:
            logger.error(f"Failed to load transformer model '{config.model_name}': {str(e)}")
            raise ModelLoadingError(f"Failed to load transformer model '{config.model_name}': {str(e)}")

    def _open_pdf(self, pdf_input: Union[str, bytes]) -> fitz.Document:
        """
        Open a PDF document.
        :param pdf_input: A file path (str) or PDF bytes.
        :return: A fitz.Document instance.
        """
        try:
            if isinstance(pdf_input, bytes):
                return fitz.open(stream=pdf_input, filetype="pdf")
            else:
                return fitz.open(pdf_input)
        except Exception as e:
            logger.error(f"PDF opening failed: {str(e)}")
            raise PDFExtractionError(f"Failed to open PDF: {str(e)}")

    def extract_text_from_pdf(self, pdf_input: Union[str, bytes]) -> str:
        """Extract text from a PDF safely."""
        try:
            with self._open_pdf(pdf_input) as doc:
                text = " ".join(page.get_text() for page in doc)
                logger.debug("Extracted text length: %d", len(text))
                return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}")

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences. Uses basic punctuation and newline heuristics.
        Only returns sentences longer than the minimum sentence length.
        """
        # List of sentence terminators (includes some non-ASCII if needed).
        terminators = ['。', '？', '!', '\n']
        # Normalize double newlines and duplicate terminators.
        text = text.replace('\n\n', '。').replace('。。', '。')
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in terminators:
                sentence = ''.join(current).strip()
                if sentence and len(sentence) > self.config.min_sentence_length:
                    sentences.append(sentence)
                current = []
        # Add any trailing text as a sentence.
        if current:
            sentence = ''.join(current).strip()
            if sentence and len(sentence) > self.config.min_sentence_length:
                sentences.append(sentence)
        logger.debug("Total sentences split: %d", len(sentences))
        return sentences

    def compute_semantic_similarity(self, summary: str, sentences: List[str]) -> NDArray:
        """Compute semantic similarity between a summary and each sentence."""
        try:
            # Break the summary into sentences.
            summary_sentences = self.split_into_sentences(summary) if isinstance(summary, str) else summary
            # Preprocess both summary sentences and document sentences.
            summary_sentences_proc = [self.text_processor.preprocess_text(sent) for sent in summary_sentences]
            processed_sentences = [self.text_processor.preprocess_text(sent) for sent in sentences]

            # Get embeddings.
            summary_embeddings = self.model.encode(
                summary_sentences_proc,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            sentence_embeddings = self.model.encode(
                processed_sentences,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            # Convert tensor embeddings to numpy arrays.
            summary_embeddings = summary_embeddings.cpu().numpy()
            sentence_embeddings = sentence_embeddings.cpu().numpy()

            # Weight summary sentences by their length.
            weights = np.array([len(sent.split()) for sent in summary_sentences_proc], dtype=np.float32)
            weights /= weights.sum()
            # Compute a weighted average embedding for the summary.
            avg_summary_embedding = np.average(summary_embeddings, axis=0, weights=weights)
            avg_summary_embedding /= np.linalg.norm(avg_summary_embedding)

            # Compute cosine similarity (dot product, since embeddings are normalized).
            similarities = np.dot(sentence_embeddings, avg_summary_embedding)
            # Apply a sigmoid to scale similarity scores between 0 and 1.
            scaled_similarities = 1 / (1 + np.exp(-5 * (similarities - 0.5)))
            return scaled_similarities
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise DocumentProcessingError(f"Similarity computation failed: {str(e)}")

    def highlight_pdf(self, pdf_input: Union[str, bytes], phrases: List[str]) -> bytes:
        """
        Add highlight annotations to the PDF for each phrase.
        Returns the highlighted PDF as bytes.
        """
        try:
            # Open the input PDF.
            doc = self._open_pdf(pdf_input)
            # Iterate through each page and highlight found phrases.
            for page in doc:
                for phrase in phrases:
                    # Search for the phrase on the page.
                    for inst in page.search_for(phrase):
                        page.add_highlight_annot(inst)
            # Save the updated PDF to a temporary file, then read its bytes.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                temp_filename = tmp_file.name
            try:
                doc.save(temp_filename)
                with open(temp_filename, "rb") as f:
                    highlighted_pdf_bytes = f.read()
            finally:
                Path(temp_filename).unlink(missing_ok=True)
            logger.debug("Highlighted PDF generated successfully.")
            return highlighted_pdf_bytes
        except Exception as e:
            logger.error(f"PDF highlighting failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to highlight PDF: {str(e)}")


def highlight_pdf_document(
    pdf_input: Union[str, bytes],
    summary: str,
    config: Optional[Config] = None
) -> bytes:
    """
    Process a PDF document to highlight key sentences based on a summary.
    
    :param pdf_input: Either a file path (str) to a PDF or PDF data as bytes.
    :param summary: The summary text based on which sentences are highlighted.
    :param config: Optional configuration overrides.
    :return: The highlighted PDF as bytes.
    :raises DocumentProcessingError: If processing fails at any stage.
    """
    config = config or Config()
    start_time = time.time()
    logger.info("Starting document processing.")

    try:
        processor = DocumentProcessor(config)
        # Extract text from the PDF.
        text = processor.extract_text_from_pdf(pdf_input)
        # Split extracted text into sentences.
        sentences = processor.split_into_sentences(text)
        # Compute semantic similarity between the summary and each sentence.
        similarities = processor.compute_semantic_similarity(summary, sentences)
        # Select sentences that have a similarity score above the threshold.
        important_phrases = [
            sent for sent, score in zip(sentences, similarities)
            if score > config.similarity_threshold
        ]
        logger.info("Found %d important phrase(s) to highlight.", len(important_phrases))
        # Highlight the important phrases in the PDF.
        highlighted_pdf = processor.highlight_pdf(pdf_input, important_phrases)
        processing_time = time.time() - start_time
        logger.info("Processing completed in %.2f seconds", processing_time)
        return highlighted_pdf
    except DocumentProcessingError as e:
        logger.error("Document processing failed: %s", str(e))
        raise

# The module is now ready for import and use by a Flask app.
# For example, in your Flask route you could use:
#
#    from highlight_key_terms import highlight_pdf_document
#
#    @app.route("/highlight", methods=["POST"])
#    def highlight():
#        pdf_file = request.files["pdf"].read()
#        summary = request.form["summary"]
#        highlighted_pdf = highlight_pdf_document(pdf_file, summary)
#        return send_file(io.BytesIO(highlighted_pdf), mimetype="application/pdf", as_attachment=True, attachment_filename="highlighted.pdf")
#
# Remove or modify any __main__ testing code as needed.
