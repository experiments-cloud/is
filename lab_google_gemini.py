"""
State Transition Stress Test - Google Gemini Evaluation
-------------------------------------------------------
Description:
    Evaluates the capacity of Google's Gemini architectures (e.g., Gemini 3.1 Pro, 
    Gemini 2.5 Flash) to maintain dynamic topological state tracking over extended 
    autoregressive inference chains.
    
Usage:
    Ensure the 'GEMINI_API_KEY' environment variable is set before execution.
"""

import os
import json
import time
import random
import csv
import re
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError

class StateTrackingExperiment:
    """Generates and evaluates deterministic state transition trajectories."""
    
    def __init__(self):
        self.registers = ["R1", "R2", "R3", "R4", "R5"]
        
    def generate_task(self, L: int) -> tuple:
        """
        Generates a sequence of deterministic algorithmic operations.
        
        Args:
            L (int): Task depth / sequence length.
            
        Returns:
            tuple: A list of string operations and a dictionary of the ground truth state.
        """
        state = {reg: 0 for reg in self.registers}
        operations = []
        for _ in range(L):
            op = random.choice(["ADD", "SUB", "MOV"])
            tgt = random.choice(self.registers)
            if op == "ADD":
                v = random.randint(1, 10)
                state[tgt] += v
                operations.append(f"Add {v} to {tgt}")
            elif op == "SUB":
                v = random.randint(1, 10)
                state[tgt] -= v
                operations.append(f"Subtract {v} from {tgt}")
            elif op == "MOV":
                src = random.choice([r for r in self.registers if r != tgt])
                state[tgt] = state[src]
                operations.append(f"Copy the current value of {src} into {tgt}")
        return operations, state
        
    def format_prompt(self, operations: list) -> str:
        """
        Constructs the neutral prompt enforcing CoT via <scratchpad> tags.
        """
        prompt = (
            "You are a deterministic state tracking system. You manage 5 memory registers: "
            "R1, R2, R3, R4, R5, all initialized to 0. Execute these operations step-by-step:\n\n"
        )
        for i, o in enumerate(operations): 
            prompt += f"{i+1}. {o}\n"
            
        prompt += (
            "\nFirst, use <scratchpad> tags to track register states after every step. "
            "Then, report the final values strictly inside <final_state> tags in valid JSON.\n"
            "Example:\n<final_state>\n{\"R1\": 0, \"R2\": 0, \"R3\": 0, \"R4\": 0, \"R5\": 0}\n</final_state>"
        )
        return prompt
        
    def evaluate_response(self, true_state: dict, llm_response: str) -> int:
        """
        Parses the model's output and applies the draconian binary success metric.
        
        Args:
            true_state (dict): The ground truth state vector.
            llm_response (str): The raw text output from the LLM.
            
        Returns:
            int: 1 if absolute match, 0 if Logical Collision (or format failure).
        """
        try:
            match = re.search(r'<final_state>(.*?)</final_state>', llm_response, re.DOTALL)
            if not match: 
                return 0
            llm_state = json.loads(match.group(1).strip())
            return 1 if all(llm_state.get(r) == true_state.get(r) for r in self.registers) else 0
        except Exception: 
            return 0

if __name__ == "__main__":
    # Initialize the Google GenAI SDK using environment variables
    api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    client = genai.Client(api_key=api_key)
    
    exp = StateTrackingExperiment()
    output_file = f"results_google_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    with open(output_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Model", "L", "Iter", "Success"])
    
    target_models = ["gemini-3.1-pro-preview", "gemini-2.5-flash"]
    
    for L in [10, 20, 50, 100]:
        for i in range(30):
            ops, truth = exp.generate_task(L)
            prompt = exp.format_prompt(ops)
            
            for model_name in target_models:
                score = "Error"
                retries = 0
                max_retries = 3
                
                while retries < max_retries:
                    try:
                        res = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.0
                            )
                        )
                        score = exp.evaluate_response(truth, res.text)
                        break # Success, exit retry loop
                        
                    except APIError as e:
                        # Handle HTTP 429 (Too Many Requests) / Quota Limits
                        if e.code == 429 or "429" in str(e):
                            wait_time = 20 * (retries + 1)
                            print(f"[!] Quota limit reached. Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            retries += 1
                        else:
                            print(f"API Error with {model_name} (L={L}, Iter={i+1}): {e}")
                            break
                            
                    except Exception as e:
                        print(f"System Error with {model_name} (L={L}, Iter={i+1}): {e}")
                        break
                
                # Safely log results
                with open(output_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([model_name, L, i+1, score])
                
                print(f"[{model_name}] L={L} | Iter={i+1} | Score={score}")
                time.sleep(2) # Throttle to respect rate limits