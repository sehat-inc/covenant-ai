import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        page.update()
    doc.save(output_pdf)



#NOTE : Change the summary text to your lease summary text
summary = "Your lease summary text here"
phrases = process_lease_contract("input.pdf", "output_highlighted.pdf", summary)