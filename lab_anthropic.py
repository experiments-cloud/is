"""
State Transition Stress Test - Anthropic Claude
-------------------------------------------------------
Description:
    Evaluates strongly aligned frontier models (Claude Sonnet) to measure the 
    impact of the "Logical Alignment Tax" on deterministic state transitions.
    
Usage:
    Ensure the 'ANTHROPIC_API_KEY' environment variable is set before execution.
"""

import os
import json
import time
import random
import csv
import re
from datetime import datetime
import anthropic

class StateTrackingExperiment:
    """Generates and evaluates deterministic state transition trajectories."""
    
    def __init__(self):
        self.registers = ["R1", "R2", "R3", "R4", "R5"]
        
    def generate_task(self, L: int) -> tuple:
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
        try:
            match = re.search(r'<final_state>(.*?)</final_state>', llm_response, re.DOTALL)
            if not match: 
                return 0
            llm_state = json.loads(match.group(1).strip())
            return 1 if all(llm_state.get(r) == true_state.get(r) for r in self.registers) else 0
        except Exception: 
            return 0

if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")
    client = anthropic.Anthropic(api_key=api_key)
    
    exp = StateTrackingExperiment()
    output_file = f"results_anthropic_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    with open(output_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Model", "L", "Iter", "Success"])
    
    target_models = ["claude-sonnet-4-6"]
    
    for L in [10, 20, 50, 100]:
        for i in range(30):
            ops, truth = exp.generate_task(L)
            prompt = exp.format_prompt(ops)
            
            for model in target_models:
                try:
                    res = client.messages.create(
                        model=model, 
                        max_tokens=1500, 
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    txt = res.content[0].text
                    score = exp.evaluate_response(truth, txt)
                    
                except Exception as e:
                    print(f"Error {model} (L={L}, Iter={i+1}): {e}")
                    score = "Error"
                
                with open(output_file, mode='a', newline='') as f:
                    csv.writer(f).writerow([model, L, i+1, score])
                    
                print(f"[{model}] L={L} | Iter={i+1} | Score={score}")
                time.sleep(3)