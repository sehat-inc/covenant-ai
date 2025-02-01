"""
Author: Hamza Amin
Dated: 1st Feb 2025
Purpose: Highlight key terms in a PDF document based on a summary.
"""


import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#NOTE: WHAT THE HELL IS GOING ON HERE COSINE WAS THE BASE FOR THIS MODULE BUT YOU HAVE EXCLUDED IT
import numpy as np
import time
import re
import logging
from pathlib import Path
from typing import List, Set, Union, Optional
from dataclasses import dataclass
from nltk.corpus import stopwords
from numpy.typing import NDArray

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
        self.stopwords: Set[str] = set(stopwords.words('english'))
        self.legal_important_words: Set[str] = {
            'shall', 'must', 'will', 'not', 'no', 'nor', 'any', 'all', 'none',
            'may', 'might', 'can', 'cannot', 'should', 'would', 'hereby'
        }
        self.stopwords -= self.legal_important_words

    def preprocess_text(self, text: str) -> str:
        """Preprocess text while preserving legal terms."""
        try:
            text = text.lower()
            text = re.sub(r'[^\w\s.]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            words = text.split()
            words = [word for word in words if word not in self.stopwords 
                    or word in self.legal_important_words]
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
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadingError(f"Failed to load model: {str(e)}")

    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from PDF safely."""
        try:
            with fitz.open(pdf_path) as doc:
                return " ".join(page.get_text() for page in doc)
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {str(e)}")
            raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}")

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with validation."""
        terminators = ['。', '？', '!', '？', '\n']
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
        
        if current:
            sentence = ''.join(current).strip()
            if sentence and len(sentence) > self.config.min_sentence_length:
                sentences.append(sentence)
        
        return sentences

    def compute_semantic_similarity(self, summary: str, sentences: List[str]) -> NDArray:
        """Compute semantic similarity with improved error handling."""
        try:
            summary_sentences = (self.split_into_sentences(summary) 
                               if isinstance(summary, str) else summary)
            
            summary_sentences = [self.text_processor.preprocess_text(sent) 
                               for sent in summary_sentences]
            processed_sentences = [self.text_processor.preprocess_text(sent) 
                                 for sent in sentences]

            summary_embeddings = self.model.encode(
                summary_sentences, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            sentence_embeddings = self.model.encode(
                processed_sentences, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )

            summary_embeddings = summary_embeddings.cpu().numpy()
            sentence_embeddings = sentence_embeddings.cpu().numpy()

            weights = np.array([len(sent.split()) for sent in summary_sentences])
            weights = weights / weights.sum()
            avg_summary_embedding = np.average(summary_embeddings, axis=0, weights=weights)
            avg_summary_embedding = avg_summary_embedding / np.linalg.norm(avg_summary_embedding)

            similarities = np.dot(sentence_embeddings, avg_summary_embedding)
            return 1 / (1 + np.exp(-5 * (similarities - 0.5)))

        except Exception as e:
            logger.error(f"Similarity computation failed: {str(e)}")
            raise DocumentProcessingError(f"Similarity computation failed: {str(e)}")

    def highlight_pdf(self, input_pdf: Union[str, Path], 
                     output_pdf: Union[str, Path], 
                     phrases: List[str]) -> None:
        """Highlight PDF with error handling."""
        try:
            with fitz.open(input_pdf) as doc:
                for page in doc:
                    for text in phrases:
                        matches = page.search_for(text)
                        for inst in matches:
                            page.add_highlight_annot(inst)
                doc.save(output_pdf)
        except Exception as e:
            logger.error(f"PDF highlighting failed: {str(e)}")
            raise DocumentProcessingError(f"Failed to highlight PDF: {str(e)}")

def process_document(input_pdf: Union[str, Path], 
                    output_pdf: Union[str, Path], 
                    summary: str, 
                    config: Optional[Config] = None) -> List[str]:
    """Main processing function with timing and logging."""
    config = config or Config()
    start_time = time.time()
    logger.info(f"Starting document processing for {input_pdf}")
    
    try:
        processor = DocumentProcessor(config)
        text = processor.extract_text_from_pdf(input_pdf)
        sentences = processor.split_into_sentences(text)
        similarities = processor.compute_semantic_similarity(summary, sentences)
        
        important_phrases = [
            sent for sent, score in zip(sentences, similarities)
            if score > config.similarity_threshold
        ]
        
        processor.highlight_pdf(input_pdf, output_pdf, important_phrases)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        return important_phrases
    
    except DocumentProcessingError as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    #NOTE: NEED CHANGES HERE
    try:
        with open(r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\summarized\summary.txt", "r") as file:
            summary = file.read().strip()
        
        input_pdf = r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\raw\testPDF.pdf"
        output_pdf = "testPDF_highlighted.pdf"
        
        phrases = process_document(input_pdf, output_pdf, summary)
        logger.info(f"Successfully processed document with {len(phrases)} highlighted phrases")
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
