import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import json
import re

class GeminiAgent:
    """
    A class to interact with the Gemini API for text generation and token counting.
    """

    def __init__(self, api_key, model_name="gemini-1.5-flash"):
        """
        Initializes the GeminiAgent with an API key and model name.

        Args:
            api_key (str): The API key for accessing the Gemini API.
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-1.5-flash".
        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError("API key cannot be empty. Please set your GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text using the Gemini API.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The total number of tokens in the text.

        Raises:
            Exception: If there is an error while counting tokens from Gemini API.
        """
        try:
            response = self.model.count_tokens(text)
            return response.total_tokens
        except Exception as e:
            raise Exception(f"Gemini Token Count Error: {e}")

    def compare_summaries(self, summary1: str, summary2: str) -> dict:
        """
        Compares two summaries using the Gemini API and enforces JSON output based on a detailed prompt with schema example.
        """
        prompt_template = """
                Dear AI Assistant,
                I need you to act as an expert Contract Analyst specializing in lease agreement comparisons.
                I have two lease summaries that need to be meticulously compared. Please analyze them with extreme attention to detail and output the result as a JSON object.

                Contract 1:
                {summary1}

                Contract 2:
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

                For each category, provide:
                1. Exact differences with specific details.
                2. Which agreement has more favorable terms (Contract 1, Contract 2, or Neutral).
                3. Specific examples where terms differ.
                4. Note any missing information that should be clarified.
                5. Flag any potentially problematic clauses or conditions.

                **IMPORTANT INSTRUCTIONS FOR RESPONSE FORMATTING:**

                Your response MUST be exclusively in valid JSON format.
                Do NOT include any text, explanations, or comments outside of the JSON object.
                Do NOT enclose the JSON object in Markdown code blocks (like `json ... `).
                The output should be **pure JSON only**, and nothing else.

                Adhere strictly to the following JSON schema:

                ```json
                {{
                    "FinancialAnalysis": {{
                        "differences": [],
                        "favorableAgreement": "Contract 1" or "Contract 2" or "Neutral",
                        "concernPoints": [],
                        "missingInformation": [],
                        "recommendations": []
                    }},
                    "LeaseTerms": {{ ... }},
                    "TerminationProvisions": {{ ... }},
                    "ObligationsComparison": {{ ... }},
                    "SpecialProvisions": {{ ... }},
                    "RiskAssessment": {{ ... }},
                    "OverallRecommendation": {{
                        "summary": "...",
                        "agreementRecommendation": "Contract 1" or "Contract 2" or "Neutral",
                        "keyTakeaways": []
                    }}
                }}
                ```
                CRITICAL: Your response must be a single, valid JSON object without any markdown formatting or additional text.
                Do not include ```json or ``` markers. Return only the JSON object itself.
            """

        prompt = prompt_template.format(summary1=summary1, summary2=summary2)

        try:
            summary1_tokens = self.count_tokens(summary1)
            summary2_tokens = self.count_tokens(summary2)
            prompt_tokens = self.count_tokens(prompt)

            print(f"Summary 1 Tokens: {summary1_tokens}")
            print(f"Summary 2 Tokens: {summary2_tokens}")
            print(f"Prompt Tokens: {prompt_tokens}")
            print(f"Total Tokens: {summary1_tokens + summary2_tokens + prompt_tokens}")

            response = self.model.generate_content(prompt)
            gemini_response_text = response.text

            # Enhanced JSON parsing with better cleanup
            try:
                # Remove markdown code blocks if present
                cleaned_text = re.sub(r'```json\s*|\s*```', '', gemini_response_text)
                # Remove any leading/trailing whitespace
                cleaned_text = cleaned_text.strip()
                
                # Parse the cleaned JSON
                analysis_json = json.loads(cleaned_text)
                return analysis_json
                
            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {je}")
                # Attempt alternative cleanup if initial parsing fails
                try:
                    # More aggressive cleanup
                    cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)  # Remove non-printable chars
                    cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)     # Remove extra newlines
                    analysis_json = json.loads(cleaned_text)
                    return analysis_json
                except json.JSONDecodeError:
                    print("Failed to parse JSON even after cleanup. Returning raw response.")
                    return {
                        "error": "Failed to parse JSON response",
                        "raw_response": gemini_response_text
                    }

        except Exception as e:
            raise Exception(f"Gemini API request failed: {e}")


def main():
    """
    Main function to demonstrate and test the GeminiAgent module.
    Loads summaries from text files, initializes the GeminiAgent,
    performs summary comparison, and prints the results in JSON format.
    """
    # NOTE: get summaries from txt files - Update file paths as needed for your system
    summary1_path = r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\summarized\summary_Extract3.txt" # r prefix for raw string, avoid backslash issues
    summary2_path = r"C:\Users\mh407\OneDrive\Documents\HackaThon\covenant-ai\rag\data\summarized\summary_Extract4.txt"

    try:
        with open(summary1_path, "r", encoding="utf-8") as f1, open(summary2_path, "r", encoding="utf-8") as f2: # Explicit encoding for robustness
            summary1 = f1.read()
            summary2 = f2.read()
    except FileNotFoundError:
        print(f"Error: Summary files not found at specified paths.\nEnsure files exist at:\n- {summary1_path}\n- {summary2_path}")
        return
    except Exception as e:
        print(f"Error reading summary files: {e}")
        return


    # Initialize Gemini agent (same)
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found...")
        return
    agent = GeminiAgent(api_key)

    start_time = time.time()
    try:
        print("Performing comparison of the two summaries...\n")
        comparison_result = agent.compare_summaries(summary1, summary2)

        output_data = {
            "summary1_filename": os.path.basename(summary1_path),
            "summary2_filename": os.path.basename(summary2_path),
            "comparison_result": comparison_result
        }

        # Pretty print the JSON with proper encoding
        with open("comparison_output.json", "w", encoding="utf-8") as outfile:
            json.dump(output_data, outfile, indent=4, ensure_ascii=False)

        print("\nComparison completed successfully.")
        
        if "error" in comparison_result:
            print(f"\nWarning: {comparison_result['error']}")
            print("Check comparison_output.json for the raw response.")
        else:
            print("Results have been written to comparison_output.json")

    except Exception as e:
        print(f"Fatal error during comparison process: {e}")
        # Write error to output file
        error_output = {
            "error": str(e),
            "summary1_filename": os.path.basename(summary1_path),
            "summary2_filename": os.path.basename(summary2_path)
        }
        with open("comparison_output.json", "w", encoding="utf-8") as outfile:
            json.dump(error_output, outfile, indent=4)
    finally:
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()