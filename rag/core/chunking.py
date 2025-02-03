import numpy as np 
import re  
from functools import lru_cache
from typing import List
import logging
import time
import sys
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", 
                    handlers=[logging.StreamHandler(sys.stdout)], force=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticChunker: 
    def __init__(self, model=model, min_tokens: int = 100, max_tokens: int = 500, buffer_size: int = 1):
        self.model = model
        self.sentence_split_pattern = re.compile(r'(?<=[.?!])(?:\s+|\n)')
        self.batch_size = 16
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size

    @lru_cache(maxsize=1024)
    def split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in self.sentence_split_pattern.split(text) if s.strip()]

    def combined_sentences_batch(self, sentences: List[str]) -> List[str]:
        n = len(sentences)
        combined = []
        for i in range(n):
            parts = [
                sentences[j]
                for j in range(i - self.buffer_size, i + self.buffer_size + 1)
                if 0 <= j < n
            ]
            combined.append(" ".join(parts))
        return combined

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        # Directly use the SentenceTransformer model to get embeddings
        return self.model.encode(texts, convert_to_numpy=True)

    def _calculate_distances_vectorized(self, embeddings: np.ndarray) -> np.ndarray:
        """Vectorized distance calculation"""
        # Calculate similarities between consecutive embeddings
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        # Convert to distances
        return 1 - similarities
    
    def group_candidates(self, sentences: List[str], breakpoint_indices: List[int]) -> List[List[str]]:
        """
        Slice the list of sentences into groups based on breakpoints.
        """
        groups = []
        start = 0
        for idx in breakpoint_indices:
            groups.append(sentences[start: idx + 1])
            start = idx + 1
        if start < len(sentences):
            groups.append(sentences[start:])
        return groups

    def group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Greedily group sentences so that each chunk's token count is within max_tokens.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        for s in sentences:
            token_count = len(s.split())
            if current_chunk and current_tokens + token_count > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_tokens = token_count
            else:
                current_chunk.append(s)
                current_tokens += token_count
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that have too few tokens (i.e. below min_tokens) with the preceding chunk
        if the merge does not exceed max_tokens.
        """
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            token_count = len(chunk.split())
            prev_token_count = len(merged[-1].split())
            if token_count < self.min_tokens and prev_token_count + token_count <= self.max_tokens:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)
        return merged
    
    def chunk_text(self, text: str, percentile_threshold: float = 80) -> List[str]:
        """
        Chunk text with optimized processing
        Args:
            text: Input text to chunk
            percentile_threshold: Percentile threshold for chunk boundaries (default: 80)
        Returns:
            List of text chunks
        """
        print("DEBUG 1")

        logging.info("Started chunking process")

        start_time = time.time()

        #Step1 
        t1 = time.time()
        # Split and combine sentences
        single_sentences = self.split_sentences(text)
        if len(single_sentences) <= 1:
            return single_sentences
        logging.info(f"Sentence Splitting: {time.time() - t1:.4f}")

        #Step 2
        t2 = time.time()
        combined_sentences = self.combined_sentences_batch(single_sentences)
        logging.info(f"Sentence Combined: {time.time() - t2:.4f}")

        #Step 3 
        t3 = time.time()
        # Get embeddings and calculate distances
        #embeddings = self.get_embeddings_batch(combined_sentences)
        embeddings = self.get_embeddings(combined_sentences)
        logging.info(f"Embeddings: {time.time() - t3:.4f}")

        #Step 4 
        t4 = time.time()
        distances = self._calculate_distances_vectorized(embeddings)
        logging.info(f"Vectorized Distances: {time.time() - t4:.4f}")

        #Step 5
        t5 = time.time()
        # Find breakpoints
        threshold = np.percentile(distances, percentile_threshold)
        indices_above_thresh = np.where(distances > threshold)[0].tolist()
        logging.info(f"Breakpoint: {time.time() - t5:.4f}")
        
        #Step 6
        t6 = time.time()
        # Create chunks efficiently

        candidate_groups = self.group_candidates(single_sentences, indices_above_thresh)

        logging.info(f"Candidate chunking: {time.time() - t6:.4f} seconds")

        # Step 7: Process candidate chunks:
        #         - If a candidate chunk exceeds max_tokens, split it further.
        final_chunks = []
        for group in candidate_groups:
            group_text = " ".join(group)
            if len(group_text.split()) > self.max_tokens:
                final_chunks.extend(self.group_sentences(group))
            else:
                final_chunks.append(group_text)

        # Step 8: Merge any chunks that are too small.
        merged_chunks = self.merge_small_chunks(final_chunks)
        
        logging.info(f"Chunking Generation: {time.time() - t6:.4f}")

        logging.info(f"Total Time: {time.time() - start_time:.4f}")
        
        return merged_chunks

def read_text_file(file_path: str) -> str:
    """Reads a text file and returns its content as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():

    file_path = "/home/ali-vijdaan/Projects/covenant-ai/rag/data/processed/Extract4_extracted.txt"
    text = read_text_file(file_path)

    chunker = SemanticChunker(model=model, min_tokens=100, max_tokens=1024)
    chunked_text = chunker.chunk_text(text=text)

    return chunked_text

if __name__ == "__main__":
    main()
