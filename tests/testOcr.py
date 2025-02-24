import pytesseract
from PIL import Image
import fitz
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
def process_scanned_page(page) -> str:
        """
        Process scanned pages using OCR

        Args:
            page: PDF page object
        Returns:
            str: Extracted text from the scanned page
        """
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="eng")
            return text
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            return ""
        
page = fitz.open(r"c:\Users\mh407\Downloads\Biology-9 BWP-18.pdf")

# Iterate through each page in the PDF
for page_number in range(page.page_count):
    page_content = page.load_page(page_number)
    text = process_scanned_page(page_content)
    print(f"Page {page_number + 1}: {text}")