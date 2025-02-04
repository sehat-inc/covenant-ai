from pdfExtractor import extract_from_pdf
import time
start = time.time()
result = extract_from_pdf(r"c:\Users\pc\Downloads\Documents\sample-pdf-download-10-mb.pdf")
end = time.time()
# print(result)
print(result['pages'])
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
# print(result['tables'])
print("Time Taken: ", end-start)
