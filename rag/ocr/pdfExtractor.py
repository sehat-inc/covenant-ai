"""
Module: pdfExtractor.py

Description:
    This module extracts text and tables from a PDF lease agreement with low latency.
    The output is designed for further processing by an LLM.
    
Usage:
    from pdf_extractor import extract_from_pdf
    result = extract_from_pdf("path/to/lease_agreement.pdf")
    # result['pages'] -> list of pages with text
    # result['tables'] -> list of tables (each table is a list of rows)
"""

import pdfplumber
import fitz  # PyMuPDF
import re
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TableData:
    content: List[List[str]]
    page_number: int
    location: Tuple[float, float, float, float]  # Not used, default value provided


class PDFExtractor:
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF extractor.

        Args:
            pdf_path: Path to the PDF file.
        """
        self.pdf_path = Path(pdf_path)
        self.extracted_text: List[Dict] = []  # List to hold text from each page
        self.tables: List[TableData] = []     # List to hold extracted tables

    def _check_for_scanned_content(self, page) -> bool:
        """
        Determine if a page is likely scanned (image-based) by checking text length.

        Args:
            page: PyMuPDF page object.
        Returns:
            bool: True if the page appears to be scanned.
        """
        text = page.get_text()
        # If text is missing or very short, assume it's scanned.
        return not text or len(text.strip()) < 50

    def _process_scanned_page(self, page) -> str:
        """
        Process a scanned page using OCR.

        Args:
            page: PyMuPDF page object.
        Returns:
            str: Text extracted using OCR.
        """
        try:
            # Increase resolution for better OCR accuracy.
            zoom = 300 / 72  # 300 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng")
            return text
        except Exception as e:
            logger.error(f"OCR processing error on page {page.number}: {e}")
            return ""

    def _extract_tables(self, plumber_page) -> List[TableData]:
        """
        Extract tables from a pdfplumber page.

        Args:
            plumber_page: pdfplumber page object.
        Returns:
            List[TableData]: List of tables extracted from the page.
        """
        tables = []
        try:
            for table in plumber_page.extract_tables():
                if table:
                    # Clean each cell in the table.
                    processed_table = [
                        [str(cell).strip() if cell is not None else "" for cell in row]
                        for row in table
                    ]
                    # Filter out empty rows.
                    processed_table = [
                        row for row in processed_table if any(cell for cell in row)
                    ]
                    if processed_table:
                        tables.append(TableData(
                            content=processed_table,
                            page_number=plumber_page.page_number,
                            location=(0, 0, 0, 0)  # Default bounding box value.
                        ))
        except Exception as e:
            logger.error(f"Error extracting tables on page {plumber_page.page_number}: {e}")
        return tables

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text.
        Returns:
            str: Cleaned text.
        """
        # Replace multiple spaces or tabs with a single space.
        text = re.sub(r"[ \t]+", " ", text)
        # Preserve paragraphs by converting multiple newlines.
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def extract(self) -> Dict:
        """
        Extract text and tables from the PDF.

        Returns:
            Dict: A dictionary with keys 'pages' (list of page text) and 'tables' (list of tables).
        """
        try:
            # Open the PDF with both PyMuPDF and pdfplumber.
            doc = fitz.open(self.pdf_path)
            with pdfplumber.open(self.pdf_path) as pdf:
                total_pages = len(doc)
                logger.info(f"Processing {total_pages} pages from {self.pdf_path}")
                for page_num in range(total_pages):
                    # Get pages from both libraries.
                    mupdf_page = doc[page_num]
                    plumber_page = pdf.pages[page_num]

                    # Check if the page is scanned.
                    if self._check_for_scanned_content(mupdf_page):
                        text = self._process_scanned_page(mupdf_page)
                        page_tables = []  # Skip table extraction for scanned pages.
                    else:
                        text = mupdf_page.get_text()
                        page_tables = self._extract_tables(plumber_page)

                    cleaned_text = self._clean_text(text)
                    self.extracted_text.append({
                        "page_number": page_num + 1,
                        "text": cleaned_text,
                    })
                    self.tables.extend(page_tables)
            doc.close()

            # Return the extracted pages and tables.
            # For tables, we return just the content (list of rows) for simplicity.
            return {
                "pages": self.extracted_text,
                "tables": [table.content for table in self.tables]
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise


def extract_from_pdf(pdf_path: str) -> Dict:
    """
    Convenience function to extract text and tables from a PDF.

    Args:
        pdf_path: Path to the PDF file.
    Returns:
        Dict: A dictionary containing the extracted text (per page) and tables.
    """
    extractor = PDFExtractor(pdf_path)
    return extractor.extract()


#NOTE: For testing purposes, you can run this module directly.
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        result = extract_from_pdf(pdf_file)
        # Print out the results (for debugging)
        print("Extracted Pages:")
        for page in result["pages"]:
            print(f"\n=== Page {page['page_number']} ===")
            print(page["text"])
        print("\nExtracted Tables:")
        for idx, table in enumerate(result["tables"], start=1):
            print(f"\n--- Table {idx} ---")
            for row in table:
                print(row)
    else:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
