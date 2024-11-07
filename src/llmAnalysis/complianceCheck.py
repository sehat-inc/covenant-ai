from typing import Dict
import os
import PyPDF2
from dotenv import load_dotenv

load_dotenv(r"../../.env")
import google.generativeai as genai


class LeaseComplianceChecker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.knowledge_base = []

    def loadKnowledgeBase(self, rules_file: str) -> None:
        """Load compliance rules from a text file"""
        with open(rules_file, "r") as file:
            self.knowledge_base = file.readlines()
            self.knowledge_base = [
                rule.strip() for rule in self.knowledge_base if rule.strip()
            ]

    def extractTextFromPdf(self, pdf_path: str) -> str:
        """Extract text content from PDF lease agreement"""
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def analyzeCompliance(self, lease_text: str) -> Dict:
        """Analyze lease agreement compliance using Gemini"""

        prompt = f"""
        Analyze the following lease agreement against these compliance rules:
        Rules:
        {' '.join(self.knowledge_base)}
        
        Lease Agreement:
        {lease_text}
        
        Please provide:
        1. Compliance status for each rule (Compliant/Non-compliant)
        2. Specific violations if any
        3. Overall compliance score (0-100)
        """

        try:
            response = self.model.generate_content(prompt)
            analysis_result = response.text

            # Process and structure the response
            return {"analysis": analysis_result, "status": "success"}
        except Exception as e:
            return {"analysis": None, "status": "error", "error_message": str(e)}


def main():
    api_key = os.getenv("Gemini_api_key")
    checker = LeaseComplianceChecker(api_key)

    rules_file = r"car_lease_agreement_rules.txt"
    checker.loadKnowledgeBase(rules_file)

    # change the path as required
    lease_pdf = r"C:\Users\mh407\OneDrive\Documents\HackaThon\sehatgang-lease-insights\data\raw\Long-Term-Vehicle-Lease-Agreement.pdf"

    if not os.path.exists(lease_pdf):
        print(f"File not found at {lease_pdf}")
        return

    print(f"File found at {lease_pdf}")

    lease_text = checker.extractTextFromPdf(lease_pdf)
    result = checker.analyzeCompliance(lease_text)

    if result["status"] == "success":
        print("Analysis Results:")
        print(result["analysis"])
    else:
        print(f"Error: {result['error_message']}")


if __name__ == "__main__":
    main()
