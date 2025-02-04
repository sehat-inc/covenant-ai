# rag/data/firebase/service_retrieve.py
from .firebase_config import firestore_client, bucket

def get_all_contracts():
    """Retrieve all contracts from Firestore"""
    try:
        contracts = []
        docs = firestore_client.collection('contracts').order_by('upload_date', direction='DESCENDING').stream()
        
        for doc in docs:
            contract_data = doc.to_dict()
            contracts.append(contract_data)
            
        return contracts
    
    except Exception as e:
        raise Exception(f"Error retrieving contracts: {str(e)}")

def get_contract_by_id(contract_id):
    """Retrieve specific contract by ID"""
    try:
        doc = firestore_client.collection('contracts').document(contract_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    
    except Exception as e:
        raise Exception(f"Error retrieving contract: {str(e)}")