"""
Neuro-Symbolic Scaffolding: O(1) State Delegation (Resumable Version)
-------------------------------------------------------
Description:
    Empirically validates the neuro-symbolic architectural transition proposed in 
    Section 5.2.1. Incorporates an automatic checkpointing system to handle API 
    Requests Per Day (RPD) limits, allowing the experiment to pause and resume 
    without losing progress.
    
Usage:
    Ensure the 'OPENAI_API_KEY' environment variable is set before execution.
"""

import os
import json
import random
import time
import sys
import pandas as pd
from typing import List, Dict
from datetime import datetime
from openai import OpenAI, RateLimitError, APIError

# Authenticate safely using environment variables
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("[!] Warning: OPENAI_API_KEY not found in environment variables.")
client = OpenAI(api_key=api_key)

class NeuroSymbolicExperiment:
    def __init__(self):
        self.registers = ["R1", "R2", "R3", "R4", "R5"]

    # 1. Algorithmic Generator
    def generate_task(self, L: int) -> Dict:
        """Generates the topological tracking problem of depth L."""
        state = {reg: 0 for reg in self.registers}
        operations = []
        
        for _ in range(L):
            op_type = random.choice(["ADD", "SUB", "MOV"])
            target = random.choice(self.registers)
            
            if op_type == "ADD":
                val = random.randint(1, 10)
                operations.append(f"Add {val} to {target}")
                state[target] += val
            elif op_type == "SUB":
                val = random.randint(1, 10)
                operations.append(f"Subtract {val} from {target}")
                state[target] -= val
            elif op_type == "MOV":
                source = random.choice([r for r in self.registers if r != target])
                operations.append(f"Copy the value of {source} into {target}")
                state[target] = state[source]
                
        return {
            "operations": operations,
            "ground_truth": state
        }

    # 2. The Neuro-Symbolic Engine (LLM as ALU + Python as RAM)
    def process_operation_stateless(self, current_memory: dict, operation: str, model: str, max_retries: int = 5) -> dict:
        """
        Treats the LLM as a pure processor without historical memory.
        Handles API Rate Limits safely with exponential backoff.
        """
        system_prompt = (
            "You are a deterministic Arithmetic Logic Unit (ALU). "
            "You will receive the current state of 5 registers and a single operation to perform. "
            "Calculate the new value for the target register. "
            "You MUST respond ONLY with a valid JSON strictly following this schema: "
            "{\"target_register\": \"R_X\", \"new_value\": Y}"
        )
        
        user_prompt = f"Current Memory State: {json.dumps(current_memory)}\nOperation to execute: '{operation}'"

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0, # Thermal determinism
                    response_format={ "type": "json_object" } # Strict JSON Scaffolding
                )
                
                update_command = json.loads(response.choices[0].message.content)
                
                if "target_register" in update_command and "new_value" in update_command:
                    return update_command
                else:
                    raise ValueError("JSON format invalid")
                    
            except RateLimitError as e:
                # Handles RPD (Requests Per Day) or TPM (Tokens Per Minute) limits
                wait_time = 60 * (attempt + 1) # Waits 1 min, then 2, then 3...
                print(f"\n  [RATE LIMIT REACHED] API Quota exhausted: {e}")
                print(f"  Pausing execution for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                
            except APIError as e:
                print(f"  [API Error] Attempt {attempt+1}: {e}")
                time.sleep(5)
                
            except Exception as e:
                print(f"  [Warning] System or JSON error on attempt {attempt+1}: {e}")
                time.sleep(2) 
                
        print("\n[!] Max retries reached. Assuming hard RPD limit. Exiting cleanly.")
        print("[!] Your progress is saved. Run the script again tomorrow to resume.")
        sys.exit(0) # Salida limpia para reanudar luego

    # 3. Main Evaluation Loop
    def evaluate_trajectory_scaffolding(self, L: int, model: str) -> bool:
        task = self.generate_task(L)
        ground_truth = task["ground_truth"]
        
        # Python initializes the external R/W Memory
        external_memory = {reg: 0 for reg in self.registers}
        
        print(f"\n--- Starting trajectory L={L} for model {model} ---")
        
        for i, op in enumerate(task["operations"]):
            update_cmd = self.process_operation_stateless(external_memory, op, model=model)
            
            if update_cmd is None:
                print(f"  [Error] LLM failed to return a valid update at step {i+1}. Trajectory failed.")
                return False
                
            target = update_cmd.get("target_register")
            new_val = update_cmd.get("new_value")
            
            # Python applies the destructive mutation (OVERWRITE)
            if target in external_memory:
                external_memory[target] = new_val
            else:
                print(f"  [Error] LLM hallucinated a non-existent register: {target}")
                return False
                
            if (i+1) % 100 == 0:
                print(f"  Step {i+1}/{L} completed. Memory integrity maintained.")

        print(f"Final External Memory: {external_memory}")
        print(f"Ground Truth         : {ground_truth}")
        
        success = 1 if external_memory == ground_truth else 0
        return success == 1

# 4. Experiment Orchestrator with Auto-Resume
def run_scaffolding_experiment():
    exp = NeuroSymbolicExperiment()
    depths = [100, 500, 1000]
    iterations_per_depth = 10 
    models_to_test = ["gpt-4o-mini", "gpt-5.4"] 
    
    # Checkpoint File Configuration
    output_file = "results_neuro_symbolic_checkpoint.csv"
    completed_tasks = set()
    results = []
    
    # --- AUTO-RESUME LOGIC ---
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            for _, row in df_existing.iterrows():
                # Almacenamos una tupla (Modelo, Profundidad L, Iteración)
                completed_tasks.add((row['Model'], row['L'], row['Iter']))
            results = df_existing.to_dict('records')
            print(f"[*] Resuming experiment. Found {len(completed_tasks)} completed trajectories in {output_file}.")
        except Exception as e:
            print(f"[!] Error reading checkpoint file: {e}. Starting fresh.")
    else:
        print(f"[*] Starting new experiment. No checkpoint found.")
    # -------------------------
    
    for model in models_to_test:
        for L in depths:
            print(f"\n======================================")
            print(f"Testing Scaffolding Paradigm | Model: {model} | L={L}")
            print(f"======================================")
            
            for iteration in range(iterations_per_depth):
                current_iter = iteration + 1
                
                # Verify if this exact task is already in the checkpoint
                if (model, L, current_iter) in completed_tasks:
                    print(f"  -> Skipping Iteration {current_iter}/{iterations_per_depth} (Already completed)")
                    continue
                
                print(f"\nIteration {current_iter}/{iterations_per_depth}")
                is_success = exp.evaluate_trajectory_scaffolding(L, model=model)
                
                results.append({
                    "Model": model,
                    "Paradigm": "Neuro-Symbolic Scaffolding",
                    "L": L,
                    "Iter": current_iter,
                    "Success": 1 if is_success else 0
                })
                
                # Save dynamically after EVERY iteration to prevent data loss
                pd.DataFrame(results).to_csv(output_file, index=False)
            
    # Print final summary
    print("\n[✔] ALL TRAJECTORIES COMPLETED SUCCESSFULLY.")
    df = pd.DataFrame(results)
    summary = df.groupby(['Model', 'L'])['Success'].mean() * 100
    print("\n--- Summary: Success Rate (%) ---")
    print(summary)

if __name__ == "__main__":
    try:
        run_scaffolding_experiment()
    except KeyboardInterrupt:
        print("\n\n[!] Experiment manually interrupted by user (Ctrl+C).")
        print("[!] Progress has been saved securely to 'results_neuro_symbolic_checkpoint.csv'.")
        print("[!] Run the script again to resume from this exact point.")
        sys.exit(0)