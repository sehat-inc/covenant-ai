import concurrent.futures
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import os

# Configure paths
pdf_path = r"c:\Users\mh407\Downloads\ilovepdf_merged.pdf"  # Update this path as needed
poppler_path = r"C:\Program Files\poppler-24.02.0\Library\bin"  # Adjust this path to where you installed Poppler
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust this path if necessary
)

# Add Poppler to PATH
os.environ["PATH"] += os.pathsep + poppler_path


def process_page(page_image):
    text = pytesseract.image_to_string(page_image)
    return text


def ocr_and_extract_toc(pdf_path):
    # Convert PDF pages to images
    doc = fitz.open(pdf_path)
    total_page_count = doc.page_count
    print(f"Total pages: {total_page_count}")
    doc.close()

    pages = convert_from_path(
        pdf_path,
        300,
        first_page=0,
        last_page=total_page_count,
        thread_count=1,
        poppler_path=poppler_path,
    )

    # Process pages in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = list(
            executor.map(process_page, pages)
        )  # Map process_page function to the list of pages

    # Combine OCR text
    main_text = "\n\n".join(results)

    return main_text


if __name__ == "__main__":
    import time

    startTime = time.time()
    print("@@@@@@£££££££$$$$$$")
    print(ocr_and_extract_toc(pdf_path))
    print("@@@@@@£££££££$$$$$$")
    print("The script took {0} second !".format(time.time() - startTime))
