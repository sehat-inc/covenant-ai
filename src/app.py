
import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import project modules
from rag.ocr.pdfExtractor import PDFTextExtractor
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from supabase import create_client, Client
from rag.core.summarizer import SummarizerAgent
import google.generativeai as genai
import tempfile
from datetime import datetime
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

supabase: Client = create_client(
    os.getenv('SERVICE_KEY'),
    os.getenv('ROLE_KEY')
)

# Initialize Gemini for summarization
genai.configure(api_key=os.getenv('GEMINI_API'))
model = genai.GenerativeModel("gemini-1.5-flash")
summarizer = SummarizerAgent(llm=model)

BUCKET_NAME = 'contract-files'  # Changed bucket name to be more specific

# Custom filter for datetime formatting
@app.template_filter('format_datetime')
def format_datetime(value):
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y %I:%M %p')
        except ValueError:
            return value
    return value

@app.route('/')
def index():
    # Fetch all contracts from Supabase
    response = supabase.table('Contract').select('*').order('created_at.desc').execute()
    contracts = response.data
    return render_template('index.html', contracts=contracts)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'contract' not in request.files:
        return redirect(request.url)
    
    file = request.files['contract']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Save file temporarily
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            # Extract text using OCR
            extractor = PDFTextExtractor(temp_path)
            extracted_content = extractor.extract_text()
            print("Extraction done ", datetime.now().time())
            # Get text from all pages
            all_text = "\n".join([page['text'] for page in extracted_content['text']])
            
            # Generate summary
            summary = summarizer._run(text=all_text)
            print("Summary made: ", datetime.now().time())
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{timestamp}_{secure_filename(file.filename)}"
            
            # Upload PDF to Supabase Storage
            with open(temp_path, 'rb') as f:
                file_data = f.read()
                # Upload using the raw file data
                result = supabase.storage.from_(BUCKET_NAME).upload(
                    path=file_name,
                    file=file_data,
                    file_options={"content-type": "application/pdf"}
                )
                print(f"Upload result: {result}")
            
            # Create database entry
            contract_data = {
                'created_at': datetime.now().isoformat(),
                'contract_pdf': file_name,
                'contract_summary': summary
            }
            
            insert_result = supabase.table('Contract').insert(contract_data).execute()
            print(f"Database insert result: {insert_result}")
            print(datetime.now().time())
            # Cleanup
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Error during upload: {str(e)}")
            # Cleanup on error
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return f"Error uploading file: {str(e)}", 500
    else:
        return "Invalid file type. Please upload a PDF.", 400

@app.route('/contract/<int:id>')
def view_contract(id):
   
        # Fetch contract details from Supabase
        response = supabase.table('Contract').select('*').eq('id', id).execute()
        if not response.data:
            return "Contract not found", 404
        
        contract = response.data[0]
        
        # Get public URL correctly
        # Remove any full URLs or double slashes from the filename
        filename = contract['contract_pdf'].split('/')[-1]
        contract['pdf_url'] = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        return render_template('contract.html', contract=contract)


@app.route('/download/<int:contract_id>')
def download_contract(contract_id):
    try:
        # Fetch contract details from Supabase
        response = supabase.table('Contract').select('*').eq('id', contract_id).execute()
        if not response.data:
            return "Contract not found", 404
        
        contract = response.data[0]
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # Download the file data
        data = supabase.storage.from_(BUCKET_NAME).download(contract['contract_pdf'])
        
        # Write to temporary file
        with open(temp_file.name, 'wb') as f:
            f.write(data)
        
        return send_file(
            temp_file.name,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=os.path.basename(contract['contract_pdf'])
        )
    except Exception as e:
        print(f"Error downloading contract: {str(e)}")
        return f"Error downloading contract: {str(e)}", 500
    finally:
        # Cleanup temp file
        if 'temp_file' in locals() and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == '__main__':
    app.run(debug=True)
 