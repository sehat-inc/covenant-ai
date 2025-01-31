from typing import Optional, Any
import argparse

import google.generativeai as genai
from dotenv import load_dotenv
import os 

load_dotenv()

class SummarizerAgent:
    def __init__(self,  llm: Any, role: Optional[str] = None, background: Optional[str] = None):
        """
        Initialize an AI Agent with an optional system prompt 

        Args: 
            llm: Language Model
            role: Role of the AI Agent 
            background: Background instructions for the AI agent

        """

        self.llm = llm
        
        self.messages = []

        default_role = "Contract Specialist/Contract Manager" 

        default_background = """ You are an expert Contract Specialist tasked with summarizing 
        legal lease contracts. Follow these rules: 
        1. Simplify legal jargon and complex terms
        2. Make sure to include and highlight key terms 
        3. Summarize obligations if stated 
        4. Include important dates if stated 
        5. Clarify termination clauses if stated 
        6. Explain renewal terms if stated 
        7. Pay attention to penalties and fees 
        8. Include ALL information regardless if it is confidential
        9. Summarize special provisions to ensure understanding 
        10. Organize information logically 
        11. Provide examples where you deem necessary to clarify complex clauses or obligations
        Remember to summarize the documents in mostly bullet points for more readability
        """

        self.prompt_template = f"""
        Role: {role or default_role}

        Background: {background or default_background}

        {{text}}
        
        """
    
    def _run(self, text: str) -> str:
        """
        Process and summarize the input text. 

        Args: 
            text (str): The contract text to be summarized

        Returns: 
            str: Summarized contract with key points and explanations
        """
        try: 
            prompt = self.prompt_template.format(text=text)

            response = self.llm.generate_content(prompt)

            return response.text

        except Exception as e: 
            return f"Error processing contract: {str(e)}"

def main(): 
    """
    Main function to run the contract summarizer 
    """

    parser = argparse.ArgumentParser(description="Summarize Contract Docs")
    parser.add_argument('--output_file',
                        type=str,
                        help='Path to save summary (optional)')
    args = parser.parse_args()

    try: 
        try:
            #NOTE: abs path used -> NEED fixing
            base_dir = os.path.dirname(os.path.abspath(__file__))
            txt_path = os.path.join(base_dir, '..', 'data', 'processed', 'testPDf_extracted.txt')
            
            #output_dir = os.path.join(base_dir, '..', 'data', 'summarized', '')
            #output_dir.mkdir(parents=True, exist_ok=True)

            with open(txt_path, "r", encoding='utf-8') as file:
                contract_text = file.read()
        except Exception as e:
            return f"Error reading contract file: {str(e)}"

        #NOTE: If you want  to change the llm for the agent change the 2 lines below
        genai.configure(api_key=os.getenv('GEMINI_API'))
        model = genai.GenerativeModel("gemini-1.5-flash")

        agent = SummarizerAgent(
            llm=model
        )

        try: 
            summary = agent._run(text=contract_text)

            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as file:
                    file.write(summary)
                print(f"Summary saved to: {args.output_file}")
            else: 
                #NOTE: Need Fixing -> outputs a file named summarized rather than a txt file in summarized folder
                with open('/home/ali-vijdaan/Projects/covenant-ai/rag/data/summarized/summary.txt', 'w', encoding='utf-8') as file: 
                    file.write(summary)

        except Exception as e:
            return f"Error in Agent Summarization: {str(e)}"

    except Exception as e: 
        return f"Error: {str(e)}"
    
    return 0

if __name__ == "__main__": 
    exit(main())
