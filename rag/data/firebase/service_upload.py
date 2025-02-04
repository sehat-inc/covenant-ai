# rag/data/firebase/service_upload.py
import uuid
from datetime import datetime
from .firebase_config import firestore_client, bucket
import os
def upload_files_to_firebase(original_contract, summary):
    """Upload files to Firebase Storage and store metadata in Firestore"""
    try:
        # Generate unique ID for the contract
        contract_id = str(uuid.uuid4())
        
        # Upload original contract to Storage
        contract_blob = bucket.blob(f'contracts/{contract_id}/original.pdf')
        contract_blob.upload_from_filename(original_contract)
        contract_url = contract_blob.generate_signed_url(expiration=3600)
        
        # Upload summary to Storage
        summary_blob = bucket.blob(f'contracts/{contract_id}/summary.txt')
        summary_blob.upload_from_filename(summary)
        
        # Read summary content for Firestore
        with open(summary, 'r') as f:
            summary_text = f.read()
        
        # Store metadata in Firestore
        contract_ref = firestore_client.collection('contracts').document(contract_id)
        contract_ref.set({
            'id': contract_id,
            'original_contract_url': contract_url,
            'summary_text': summary_text,
            'upload_date': datetime.now(),
            'filename': os.path.basename(original_contract)
        })
        
        return contract_id
        
    except Exception as e:
        raise Exception(f"Error uploading to Firebase: {str(e)}")