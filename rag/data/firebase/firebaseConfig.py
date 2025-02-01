
import firebase_admin
from firebase_admin import credentials, firestore, storage

FIREBASE_CREDENTIAL_PATH = 'firebase_api.json'

cred = credentials.Certificate(FIREBASE_CREDENTIAL_PATH)
firebase_admin.initialize_app(cred, {
    'storageBucket' : 'covenant-ai.firebasestorage.app'
})

firestore_client = firestore.client()

bucket = storage.bucket()