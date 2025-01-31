import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time 

import re 
from nltk.corpus import stopwords
import nltk

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

STOPWORDS = set(stopwords.words('english'))
LEGAL_IMPORTANT_WORDS = {
    'shall', 'must', 'will', 'not', 'no', 'nor', 'any', 'all', 'none',
    'may', 'might', 'can', 'cannot', 'should', 'would', 'hereby'
}
# Remove legally significant words from stopwords
STOPWORDS = STOPWORDS - LEGAL_IMPORTANT_WORDS

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_text(text):
    """
    Preprocess text with enhanced cleaning and stopword removal, preserving legal terms.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but preserve periods for sentence splitting
    text = re.sub(r'[^\w\s.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords while preserving legal terms
    words = [word for word in words if word not in STOPWORDS or word in LEGAL_IMPORTANT_WORDS]
    
    # Rejoin text
    return ' '.join(words)

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

#NOTE: WTF is going on this function. I have no idea
def compute_semantic_similarity(summary, sentences):
    """
    Compute semantic similarity between a summary and sentences using improved cosine similarity.
    
    Args:
        summary (str): The reference summary text
        sentences (list): List of sentences to compare against the summary
        
    Returns:
        numpy.ndarray: Array of similarity scores for each sentence
    """
    # Convert summary into sentences if it's not already
    if isinstance(summary, str):
        summary_sentences = split_into_sentences(summary)
    else:
        summary_sentences = summary

    summary_sentences = [preprocess_text(sent) for sent in summary_sentences]
    processed_sentences = [preprocess_text(sent) for sent in sentences]

    # Encode both summary and target sentences
    summary_embeddings = model.encode(summary_sentences, convert_to_tensor=True, normalize_embeddings=True)
    sentence_embeddings = model.encode(processed_sentences, convert_to_tensor=True, normalize_embeddings=True)
    
    # Convert to numpy for calculations
    summary_embeddings = summary_embeddings.cpu().numpy()
    sentence_embeddings = sentence_embeddings.cpu().numpy()
    
    # Calculate weighted average of summary embeddings based on sentence length
    weights = np.array([len(sent.split()) for sent in summary_sentences])
    weights = weights / weights.sum()  # Normalize weights
    avg_summary_embedding = np.average(summary_embeddings, axis=0, weights=weights)
    
    # Normalize the averaged summary embedding
    avg_summary_embedding = avg_summary_embedding / np.linalg.norm(avg_summary_embedding)
    
    # Compute similarity scores using matrix operations
    similarities = np.dot(sentence_embeddings, avg_summary_embedding)
    
    # Apply sigmoid-like scaling to spread out similarity scores
    similarities = 1 / (1 + np.exp(-5 * (similarities - 0.5)))
    
    return similarities


def get_important_phrases(text, summary, threshold=0.63):
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

summary = """
**French Master Lease and Servicing Agreement Summary**

This document, a French Master Lease and Servicing Agreement, was originally dated September 25, 2018, amended and restated on April 29, 2021, and further amended and restated on December 21, 2021.  It's a complex agreement between several parties:

* **Lessor:** RAC FINANCE SAS (French FleetCo)
* **Lessee & Servicer:** HERTZ FRANCE SAS (French OpCo)
* **Additional Lessees:**  Affiliates of French OpCo that become lessees under specific conditions (Clause 12).
* **French Security Trustee:** BNP PARIBAS TRUST CORPORATION UK LIMITED
* **Issuer Security Trustee:** BNP PARIBAS TRUST CORPORATION UK LIMITED


**Key Terms:**

* **Lease Vehicles:** Vehicles purchased or to be purchased by the Lessor and leased to the Lessees.
* **Vehicle Lease Commencement Date:** The date leasing of a specific vehicle begins. For vehicles previously leased under a terminated agreement, this is the Closing Date.  Otherwise, it's the earlier of the date funds are expended by French FleetCo or the vehicle delivery date.
* **Vehicle Term:** The period a vehicle is leased, ending on the earliest of: Disposition Date, Rejection Date (if rejected), or French Master Lease Scheduled Expiration Date (unless extended).
* **Vehicle Lease Expiration Date:** The end date of the Vehicle Term.
* **French Master Lease Scheduled Expiration Date:**  The scheduled end date of the overall lease agreement.  This can be extended.
* **Lease Expiration Date:** The end date of the overall lease agreement, the later of the final payment of French Advances and the Vehicle Lease Expiration Date for the last vehicle.
* **Rent:** Monthly payments by Lessees to the Lessor, comprised of:
    * **Monthly Base Rent:** Pro rata portion of the Depreciation Charge.
    * **Final Base Rent:** Pro rata portion of the Depreciation Charge as of the Disposition Date.
    * **Monthly Variable Rent:** Based on interest accrued on French Advances and French Carrying Charges.
    * **Other Charges:**  Include depreciation, excess mileage, excess damage, etc.
* **Depreciation Charge:**  Calculated daily by the Lessor for each Lease Vehicle.
* **Disposition Date:** The date a Lease Vehicle is sold or otherwise disposed of.
* **Rejected Vehicle:** A Lease Vehicle rejected by a Lessee within the Inspection Period due to non-conformity.
* **Payment Date:** The date rent and other payments are due.
* **Lease Event of Default:**  Breaches of the contract, including non-payment of rent, unauthorized assignments, breach of covenants, bankruptcy of Hertz or a Lessee, and invalidity of the agreement.
* **Liquidation Event:** An event that triggers the liquidation of the Lessor's assets.  (Specific events are defined in other related documents.)
* **Servicer Default:**  Breaches by the Servicer of their obligations.
* **French Security Documents:**  Documents outlining the security interests of the French Security Trustee.
* **Manufacturer Programs:** Agreements with vehicle manufacturers regarding vehicle purchase and return terms.
* **Eligible Vehicle:** A vehicle meeting specified criteria for leasing.
* **Program Vehicle:** Vehicle subject to a Manufacturer Program.
* **Non-Program Vehicle:** Vehicle not subject to a Manufacturer Program.
* **Master Lease Termination Notice:** Notice given to terminate the lease due to a default.
* **Tax Deduction:** Withholding of taxes from payments.


**Lessee Obligations:**

* Pay rent and other charges on time.
* Maintain and repair Lease Vehicles (except for major damage covered by insurance).
* Obtain and maintain required insurance (Motor Third Party Liability Cover and Public/Product Liability Cover).
* Pay all registration fees, fines, penalties, etc., related to the Lease Vehicles.
* Return Lease Vehicles by the Vehicle Lease Expiration Date (or as directed by the Servicer).
* Comply with all covenants in the agreement.
* Provide certain reports and information to the Lessor.


**Lessor Obligations:**

* Purchase or arrange for purchase of Lease Vehicles.
* Lease Lease Vehicles to Lessees.
* Maintain depreciation records.
* Pay the Servicer's fees.
* Upon default, take possession of and dispose of Lease Vehicles.


**Termination Clauses:**

* The agreement can be terminated by the Lessor or the French Security Trustee upon a Lease Event of Default or Liquidation Event (Clause 9).  A Master Lease Termination Notice must be served.
* Lessees (other than French OpCo) may resign with written notice and payment of all dues (Clause 26).


**Renewal Terms:**

* The lease of Lease Vehicles can be extended/renewed by executing a French Master Lease Extension Agreement before or within 5 business days after the French Master Lease Scheduled Expiration Date (Clause 3).


**Penalties and Fees:**

* Late payment of rent results in default interest (Clause 4).
* Failure to comply with covenants can result in termination and damages (Clause 9).
* The Lessor is not liable for most vehicle defects or damages, and lessees waive many claims (Clause 14).  Lessees bear risk and costs related to vehicle defects and damages.


**Special Provisions:**

* The Lessor maintains ownership of all Lease Vehicles (Clause 2).  This is a "location simple" lease under French law.
* The Servicer performs various administrative functions (Clause 6) and can be replaced in case of Servicer Default.
* Subleasing of vehicles is permitted under specific conditions (Clause 5).  The Servicer must be informed and the subleases terminate following a Level 1 Minimum Liquidity Test Breach.
* Recourse against RAC Finance SAS is limited (Clause 15).
* Governing law is French law, and the Tribunal de commerce de Paris has exclusive jurisdiction (Clauses 16 and 17).


**Note:**  This summary is for informational purposes only and does not constitute legal advice.  The complete agreement should be reviewed by legal counsel for a thorough understanding of the rights and obligations of all parties.  Many details and specific definitions are omitted here for brevity as some details are referenced in other documents not included in this sample.

"""

start = time.time()

phrases = process_lease_contract(r"C:\Users\pc\Desktop\covenant-ai\rag\data\raw\Extract4.pdf", "extract_4_highlightv4.pdf", summary)

end = time.time()

print(f"Total Time: {abs(start - end)}\n")
