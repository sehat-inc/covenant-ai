import firebase_admin
from firebase_admin import credentials, firestore, storage
import os

def initialize_firebase():
    """Initialize Firebase if not already initialized"""
    if not firebase_admin._apps:
        # Get the path to the Firebase credentials file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        FIREBASE_CREDENTIAL_PATH = os.path.join(current_dir, 'firebase_api.json')
        
        # Initialize Firebase
        cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'covenant-ai-c4ea0.appspot.com'
        })

    # Return clients
    return firestore.client(), storage.bucket()

# Initialize Firestore and Storage clients
firestore_client, bucket = initialize_firebase()