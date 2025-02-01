import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def compute_semantic_similarity(summary, sentences):
    # Encode summary and sentences
    summary_embedding = model.encode([summary])[0]
    sentence_embeddings = model.encode(sentences)
    
    # Compute cosine similarity
    similarities = cosine_similarity([summary_embedding], sentence_embeddings)[0]
    return similarities

def split_into_sentences(text):
    # Simple sentence splitting by common terminators
    terminators = ['。', '？', '!', '？', '\n']
    text = text.replace('\n\n', '。').replace('。。', '。')
    sentences = []
    current = []
    
    for char in text:
        current.append(char)
        if char in terminators:
            sentence = ''.join(current).strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                sentences.append(sentence)
            current = []
            
    if current:  # Add any remaining text
        sentence = ''.join(current).strip()
        if sentence and len(sentence) > 10:
            sentences.append(sentence)
    
    return sentences

def get_important_phrases(text, summary, threshold=0.5):
    # Split text into sentences
    sentences = split_into_sentences(text)
    
    # Get similarity scores
    similarity_scores = compute_semantic_similarity(summary, sentences)
    
    # Select important sentences based on similarity threshold
    important_phrases = set()
    for i, score in enumerate(similarity_scores):
        if score > threshold:
            important_phrases.add(sentences[i])
    
    return list(important_phrases)

def process_lease_contract(input_pdf, output_pdf, summary):
    text = extract_text_from_pdf(input_pdf)
    important_phrases = get_important_phrases(text, summary)
    highlight_pdf(input_pdf, output_pdf, important_phrases)
    return important_phrases

def highlight_pdf(input_pdf, output_pdf, phrases):
    doc = fitz.open(input_pdf)
    for page in doc:
        for text in phrases:
            matches = page.search_for(text)
            for inst in matches:
                page.add_highlight_annot(inst)
        # Remove or comment out the following line:
        # page.update()
    doc.save(output_pdf)



#NOTE : Change the summary text to your lease summary text
summary = """

This summary outlines key aspects of the lease contract, aiming for clarity and comprehensiveness.  Due to the highly technical nature of the original document, some legal nuances may require further professional consultation.

**I. Depreciation Charges:**

* **Program Vehicles (within Estimation Period):** Depreciation is calculated based on the "Initially Estimated Depreciation Charge" as of the relevant date.
* **Program Vehicles (outside Estimation Period):** Depreciation uses the monthly dollar amount specified in the related Manufacturer Program.
* **Non-Program Medium-Duty Trucks:** Depreciation is calculated according to Generally Accepted Accounting Principles (GAAP) and uses a percentage based on the truck's age:
    * 0-12 months: 2.75%
    * 13-24 months: 1.42%
    * >24 months: 0.58%
* **Depreciation Record:**  The specific definition is detailed in Section 4.1 of the Lease.

**II. Key Dates and Definitions:**

* **Determination Date:** Five business days before each Payment Date.
* **Direct-to-Consumer Sale:**  A sale where Hertz or its affiliate transfers vehicle title and acts as the seller, complying with consumer protection laws.
* **Disposition Date:** The date determining the vehicle's value, varying based on the circumstances:
    * **Manufacturer Repurchase:**  The "Turnback Date."
    * **Guaranteed Depreciation Program (not sold to third party):** The "Backstop Date."
    * **Sold to Third Party:** The date sale proceeds are deposited into the "Collection Account."

**III. Special Terms:**

The lease includes "Special Terms"  that vary by vehicle location, defining the lease's duration in specific states/commonwealths.

* **Illinois:** One year
* **Iowa:** Eleven months
* **Maine:** Eleven months
* **Maryland:** 180 days
* **Massachusetts:** Eleven months
* **Nebraska:** Thirty days
* **South Dakota:** Twenty-eight days
* **Texas:** 181 days
* **Vermont:** Eleven months
* **Virginia:** Eleven months
* **West Virginia:** Thirty days


**IV.  Other Key Terms & Definitions (Summary):**

* **Required Series Noteholders:** Defined in the Base Indenture.
* **Resigning Lessee:** Defined in Section 25 of the Lease.
* **SEC:** Securities and Exchange Commission.
* **Series of Notes:** Notes issued under the Base Indenture and Series Supplement.
* **Series Supplement:** Supplement to the Base Indenture.
* **Servicer:** Defined in the Lease Preamble.
* **Servicer Default:** Defined in Section 9.6 of the Lease.
* **Servicing Standard:**  Describes the expected level of service, emphasizing promptness, diligence, and industry best practices.  Failure to meet this standard must not materially adversely affect the Lessor.

**V. Missing Information:**

This summary is based solely on the provided text.  Crucial information such as lease duration (outside of the special terms), payment amounts, penalties for late payments, lessee obligations, termination clauses (besides the definition of "Resigning Lessee"), and renewal terms are not included in the extract and therefore cannot be summarized.  A complete review of the entire lease agreement is necessary to obtain this information.
"""
start = time.time()
phrases = process_lease_contract(r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\raw\testPDf.pdf", "output_highlighted.pdf", summary)
end = time.time()
print(f"Processing time: {end - start:.2f} seconds")