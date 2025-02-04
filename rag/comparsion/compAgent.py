import requests
import json
from dotenv import load_dotenv
import os

# **Correct Gemini API URL (using Google AI Gemini API)**
# You need to replace YOUR_API_KEY with your actual Gemini API key
# The URL below is for the Gemini Pro model (text-only model)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

class GeminiAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def compare_summaries(self, summary1, summary2):
        # Build a very detailed prompt for comparing the two summaries
        prompt = f"""
Dear Gemini AI,

I have two detailed summaries of vehicle lease agreements that I need to compare comprehensively. Your task is to analyze the summaries side by side and provide a thorough comparison. Below you will find each summary followed by a list of specific areas for analysis.

---------------------------
Summary 1:
{summary1}

---------------------------
Summary 2:
{summary2}

---------------------------
Please perform the following tasks:

1. **Payment Terms:**
   - Compare the monthly payment amounts, the due dates, and any variations in how payments are structured.
   - Identify if one contract offers a more favorable payment plan compared to the other.

2. **Lease Duration:**
   - Examine the total lease period for each contract.
   - Highlight any differences in renewal options, early termination provisions, or duration flexibility.

3. **Termination Clauses:**
   - Compare the conditions under which each contract allows early termination.
   - Identify any penalties, fees, or notice periods that differ between the contracts.

4. **Insurance and Maintenance Provisions:**
   - Compare the insurance requirements (comprehensive vs. minimal liability) specified.
   - Contrast the maintenance responsibilities and how major vs. minor repairs are allocated between the parties.

5. **Risk and Additional Provisions:**
   - Analyze any extra provisions or clauses that could affect the lessee’s obligations or rights.
   - Provide a risk assessment by highlighting any potentially unfavorable terms in either summary.

6. **Other Notable Differences:**
   - Identify any additional differences that might impact a lessee's decision.
   - Include any insights on clauses that do not directly fall under the categories above.

Please return your answer as a well-structured JSON object with the following keys: "PaymentTerms", "LeaseDuration", "TerminationClauses", "InsuranceAndMaintenance", "RiskAnalysis", and "OtherDifferences". Each key should map to a detailed description of the differences between the two summaries based on that aspect.

Thank you for your comprehensive and detailed analysis.
"""

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        params = {
            "key": self.api_key
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(GEMINI_API_URL, params=params, json=data, headers=headers)
        if response.status_code == 200:
            # Assuming the API returns a JSON object containing the analysis
            response_json = response.json()
            # **Important:**  Gemini API response structure is different. You need to extract the text.
            # Check the documentation for the exact response structure.
            # Assuming the text is in 'candidates[0].content.parts[0].text'
            if 'candidates' in response_json and response_json['candidates']:
                if 'content' in response_json['candidates'][0] and response_json['candidates'][0]['content']['parts']:
                    gemini_response_text = response_json['candidates'][0]['content']['parts'][0].get('text', "No text in response")

                    # **Parse the Gemini response as JSON if it's formatted as JSON**
                    try:
                        analysis_json = json.loads(gemini_response_text)
                        return analysis_json
                    except json.JSONDecodeError:
                        print("Warning: Gemini response is not valid JSON. Returning raw text response.")
                        return {"raw_response": gemini_response_text} # Return raw text if JSON parsing fails
                else:
                    return {"error": "Unexpected API response structure: content or parts missing"}
            else:
                return {"error": "Unexpected API response structure: candidates missing"}

        else:
            raise Exception(f"Gemini API request failed: {response.status_code} - {response.text}")

    def chat(self, message):
        # Build a follow-up chat prompt that references the context of our prior comparison
        prompt = f"""
Dear Gemini AI,

Following our previous comparison of the vehicle lease agreements, I have a follow-up question. Please provide a detailed answer based on your previous analysis and any additional insights you might have.

Follow-up question:
"{message}"

Please ensure your response is clear and detailed.
"""
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        params = {
            "key": self.api_key
        }
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(GEMINI_API_URL, params=params, json=data, headers=headers)
        if response.status_code == 200:
             response_json = response.json()
             if 'candidates' in response_json and response_json['candidates']:
                if 'content' in response_json['candidates'][0] and response_json['candidates'][0]['content']['parts']:
                    return {"response": response_json['candidates'][0]['content']['parts'][0].get('text', "No text in response")}
                else:
                    return {"error": "Unexpected API response structure: content or parts missing"}
             else:
                return {"error": "Unexpected API response structure: candidates missing"}
        else:
            raise Exception(f"Gemini API request failed: {response.status_code} - {response.text}")

def main():
    # Hard-coded summaries
    summary1 = (
        "The lease contract offers a monthly payment plan of $500, payable on the 1st of every month. "
        "It spans 36 months with an option to renew for an additional 12 months at a predetermined rate. "
        "Early termination is permitted with a fee equivalent to three months’ rent. The lessee is required "
        "to maintain comprehensive insurance and perform regular maintenance on the vehicle."
    )

    summary2 = (
        "This lease agreement sets the monthly payment at $520, with payments due by the 5th day of each month. "
        "The lease is valid for 36 months, but there is no renewal option available. Early termination is allowed "
        "without penalty provided a 60-day notice is given. The lessee must secure minimal liability insurance, "
        "with minor maintenance responsibilities, while major repairs are covered by the lessor."
    )

    # Initialize the Gemini agent (replace 'YOUR_GEMINI_API_KEY' with your actual API key)
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file. Please create a .env file and set GEMINI_API_KEY.")
        return

    agent = GeminiAgent(api_key)

    # Step 2: Comparison Request
    try:
        print("Performing comparison of the two summaries...\n")
        comparison_result = agent.compare_summaries(summary1, summary2)
        print("Comparison Result:")
        print(json.dumps(comparison_result, indent=4))
    except Exception as e:
        print("Error during comparison:", e)

    # Chat loop for follow-up questions
    print("\nEnter follow-up questions for Gemini AI. Type 'exit' to quit.")
    while True:
        user_input = input("Your question: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break
        try:
            chat_response = agent.chat(user_input)
            print("Gemini AI Response:")
            print(json.dumps(chat_response, indent=4))
        except Exception as e:
            print("Error during chat:", e)

if __name__ == "__main__":
    main()