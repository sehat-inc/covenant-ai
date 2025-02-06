import requests
import json
from dotenv import load_dotenv
import os


GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
)


class GeminiAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def compare_summaries(self, summary1, summary2):
        # Build a very detailed prompt for comparing the two summaries

        prompt = f"""
                Dear AI Assistant,
                I need you to act as an expert Contract Analyst specializing in lease agreement comparisons. 
                I have two lease summaries that need to be meticulously compared. Please analyze them with extreme attention to detail.

                SUMMARY 1:
                {summary1}

                SUMMARY 2:
                {summary2}

                Please perform an exhaustive analysis focusing on these specific areas:

                1. FINANCIAL TERMS ANALYSIS:
                - Compare exact monthly/annual payment amounts
                - List all fees (administrative, processing, late fees)
                - Compare security deposits
                - Analyze payment schedules and due dates
                - Compare any penalties for late payments
                - List any hidden costs or additional charges

                2. LEASE DURATION AND RENEWAL:
                - Compare initial lease terms
                - List all renewal options and conditions
                - Compare notice periods required for renewal
                - Analyze automatic renewal clauses
                - Compare lease extension possibilities
                - List any blackout periods or seasonal restrictions

                3. TERMINATION AND EXIT CONDITIONS:
                - Compare early termination penalties
                - List required notice periods
                - Compare conditions for lease breaking
                - Analyze default conditions
                - Compare cure periods
                - List any special termination rights

                4. OBLIGATIONS AND RESPONSIBILITIES:
                - Compare maintenance responsibilities
                - List insurance requirements
                - Compare utility responsibilities
                - Analyze compliance requirements
                - Compare reporting obligations
                - List any special duties or obligations

                5. SPECIAL PROVISIONS:
                - Compare any unique clauses
                - List special rights or privileges
                - Compare any modification rights
                - Analyze dispute resolution methods
                - Compare force majeure clauses
                - List any unusual restrictions or requirements

                6. RISK ASSESSMENT:
                - Identify potential risks in each agreement
                - Compare liability allocations
                - List indemnification requirements
                - Compare warranty provisions
                - Analyze potential legal exposure
                - Compare compliance requirements

                For each category above, please:
                1. List exact differences with specific details
                2. Highlight which agreement has more favorable terms
                3. Provide specific examples where terms differ
                4. Note any missing information that should be clarified
                5. Flag any potentially problematic clauses or conditions

                Format your response as a JSON object with these main keys:
                {{
                    "FinancialAnalysis": {{}},
                    "LeaseTerms": {{}},
                    "TerminationProvisions": {{}},
                    "ObligationsComparison": {{}},
                    "SpecialProvisions": {{}},
                    "RiskAssessment": {{}},
                    "OverallRecommendation": {{}}
                }}

            Under each main key, please include:
            - "differences": [list of specific differences]
            - "favorableAgreement": "Summary1" or "Summary2"
            - "concernPoints": [list of potential issues]
            - "missingInformation": [list of unclear or missing items]
            - "recommendations": [specific suggestions]

            Thank you for your thorough analysis. Please provide detailed insights and actionable recommendations.
        """

        data = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            GEMINI_API_URL, params=params, json=data, headers=headers
        )
        if response.status_code == 200:
            # Assuming the API returns a JSON object containing the analysis
            response_json = response.json()
            # **Important:**  Gemini API response structure is different. You need to extract the text.
            # Check the documentation for the exact response structure.
            # Assuming the text is in 'candidates[0].content.parts[0].text'
            if "candidates" in response_json and response_json["candidates"]:
                if (
                    "content" in response_json["candidates"][0]
                    and response_json["candidates"][0]["content"]["parts"]
                ):
                    gemini_response_text = response_json["candidates"][0]["content"][
                        "parts"
                    ][0].get("text", "No text in response")

                    # **Parse the Gemini response as JSON if it's formatted as JSON**
                    try:
                        analysis_json = json.loads(gemini_response_text)
                        return analysis_json
                    except json.JSONDecodeError:
                        print(
                            "Warning: Gemini response is not valid JSON. Returning raw text response."
                        )
                        return {
                            "raw_response": gemini_response_text
                        }  # Return raw text if JSON parsing fails
                else:
                    return {
                        "error": "Unexpected API response structure: content or parts missing"
                    }
            else:
                return {
                    "error": "Unexpected API response structure: candidates missing"
                }

        else:
            raise Exception(
                f"Gemini API request failed: {response.status_code} - {response.text}"
            )

    def chat(self, message):
        # Build a follow-up chat prompt that references the context of our prior comparison
        prompt = f"""
        Dear Legal AI Assistant,

        Following our previous comparison of the vehicle lease agreements, I have a follow-up question. 
        Please provide a detailed answer based on your previous analysis and any additional insights you might have.

        Follow-up question:
        "{message}"

        Please ensure your response is clear and detailed.
        """
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            GEMINI_API_URL, params=params, json=data, headers=headers
        )
        if response.status_code == 200:
            response_json = response.json()
            if "candidates" in response_json and response_json["candidates"]:
                if (
                    "content" in response_json["candidates"][0]
                    and response_json["candidates"][0]["content"]["parts"]
                ):
                    return {
                        "response": response_json["candidates"][0]["content"]["parts"][
                            0
                        ].get("text", "No text in response")
                    }
                else:
                    return {
                        "error": "Unexpected API response structure: content or parts missing"
                    }
            else:
                return {
                    "error": "Unexpected API response structure: candidates missing"
                }
        else:
            raise Exception(
                f"Gemini API request failed: {response.status_code} - {response.text}"
            )


def main():

    # NOTE: get summaries from txt files
    summary1 = open(
        r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\summarized\summary_Extract1.txt",
        "r",
    ).read()

    summary2 = open(
        r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\summarized\summary_Extract2.txt",
        "r",
    ).read()

    # Initialize the Gemini agent (replace 'YOUR_GEMINI_API_KEY' with your actual API key)
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY not found in .env file. Please create a .env file and set GEMINI_API_KEY."
        )
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
