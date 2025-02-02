from pdfExtractor import extract_from_pdf
import time
start = time.time()
result = extract_from_pdf(r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\raw\Extract4.pdf")
end = time.time()
print(result['pages'])
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print(result['tables'])
print("Time Taken: ", end-start)
