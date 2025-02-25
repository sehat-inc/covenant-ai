import fitz
import time
import re
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFHighlighter:
    def __init__(self, model, similarity_threshold: float = 0.63, min_sentence_length: int = 10):
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.min_sentence_length = min_sentence_length
        self.sentence_split_pattern = re.compile(r'[.!?\n]')
    
    def _open_pdf(self, pdf_input: Union[str, bytes]) -> fitz.Document:
        try:
            if isinstance(pdf_input, bytes):
                return fitz.open(stream=pdf_input, filetype="pdf")
            else:
                return fitz.open(pdf_input)
        except Exception as e:
            logger.error(f"PDF opening failed: {str(e)}")
            raise Exception(f"Failed to open PDF: {str(e)}")

    def extract_text(self, pdf_input: Union[str, bytes]) -> str:
        try:
            with self._open_pdf(pdf_input) as doc:
                return " ".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def split_sentences(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self.sentence_split_pattern.split(text) if len(s.strip()) > self.min_sentence_length]
        return sentences
    
    def compute_semantic_similarity(self, summary: str, sentences: List[str]) -> np.ndarray:
        try:
            summary_sentences = self.split_sentences(summary)
            summary_embeddings = self.model.encode(summary_sentences, convert_to_numpy=True, normalize_embeddings=True)
            sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            avg_summary_embedding = np.mean(summary_embeddings, axis=0)
            similarities = np.dot(sentence_embeddings, avg_summary_embedding)
            return 1 / (1 + np.exp(-5 * (similarities - 0.5)))
        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise Exception(f"Similarity computation failed: {str(e)}")

    def highlight_pdf(self, pdf_input: Union[str, bytes], phrases: List[str]) -> bytes:
        try:
            doc = self._open_pdf(pdf_input)
            for page in doc:
                for phrase in phrases:
                    for inst in page.search_for(phrase):
                        page.add_highlight_annot(inst)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                temp_filename = tmp_file.name
            try:
                doc.save(temp_filename)
                with open(temp_filename, "rb") as f:
                    highlighted_pdf_bytes = f.read()
            finally:
                Path(temp_filename).unlink(missing_ok=True)
            return highlighted_pdf_bytes
        except Exception as e:
            logger.error(f"PDF highlighting failed: {str(e)}")
            raise Exception(f"Failed to highlight PDF: {str(e)}")
    
    def process_pdf(self, pdf_input: Union[str, bytes], summary: str) -> bytes:
        start_time = time.time()
        logger.info("Starting document processing.")
        try:
            text = self.extract_text(pdf_input)
            sentences = self.split_sentences(text)
            similarities = self.compute_semantic_similarity(summary, sentences)
            important_phrases = [sent for sent, score in zip(sentences, similarities) if score > self.similarity_threshold]
            logger.info(f"Found {len(important_phrases)} important phrase(s) to highlight.")
            highlighted_pdf = self.highlight_pdf(pdf_input, important_phrases)
            logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
            return highlighted_pdf
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise


# if __name__ == "__main__":
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
#     highlighter = PDFHighlighter(model)
#     pdf_input = "sample.pdf"
#     summary = "This is a sample summary."
#     highlighted_pdf = highlighter.process_pdf(pdf_input, summary)
#     with open("highlighted_sample.pdf", "wb") as f:
#         f.write(highlighted_pdf) 
