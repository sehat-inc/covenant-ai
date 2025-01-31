def estimate_tokens(text): 
    return len(text.split()) * 1.1 # assumption of Google Tokenizer word to token ratio

# testing file path
# had some prob with relative path -> NOTE: Needs Fixing
with open("/home/ali-vijdaan/Projects/covenant-ai/rag/data/processed/Extract4_extracted.txt", "r", encoding='utf-8') as f:
    text = f.read()

print(f"Estimated tokens: {estimate_tokens(text)}")