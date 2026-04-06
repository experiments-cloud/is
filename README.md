# Theoretical Limits of State Tracking in Autoregressive Models

This repository contains the official replication code and synthetic evaluation environments for the paper **"Theoretical limits of state tracking in autoregressive models"**. 

The provided scripts execute the **State Transition Stress Test**, a deterministic algorithmic environment designed to isolate and evaluate the topological state-tracking capabilities of frontier Large Language Models (LLMs) across varying inference depths ($L$).

## 🚀 Repository Structure

* `lab_google_gemini.py`: Evaluates the Gemini model family (Gemini 3.1 Pro, Gemini 2.5 Flash) using the new Google GenAI SDK.
* `lab_openai.py`: Evaluates the OpenAI model family (GPT-4.1, GPT-5.4 paradigms) to measure the impact of latent reasoning vs. standard probabilistic generation.
* `lab_anthropic_claude.py`: Evaluates Anthropic's Claude 4.6 Sonnet to observe the effects of strong alignment on logical coherence.
* `lab_groq.py`: Evaluates high-density open-weight models (Llama 3.3 70B, Llama 3.1, Qwen 32B) hosted via Groq to document the Cliff Collapse phenomenon.
* `lab_mechanistic_extraction.py`: Performs a forward pass on Llama-3.1-8B to extract self-attention matrices and generates the Attention Dilution heatmap, demonstrating internal context fragmentation.
* `lab_neuro_symbolic_scaffolding.py`: Validates the proposed architectural transition (Section 5.2.1). Connects OpenAI models to a deterministic Python dictionary acting as external R/W RAM. By forcing the LLM to act as a stateless ALU, it proves that O(1) state delegation entirely bypasses attention dilution and prevents state collapse even at extreme depths (L∈{100,500,1000}).

## ⚙️ Prerequisites and Installation

This project requires **Python 3.8+**. 

Install the required official SDKs via pip:

```bash
pip install google-genai openai anthropic torch transformers huggingface_hub matplotlib seaborn pandas
```

🔐 Security and Anonymity (API Keys)

To maintain security and comply with best practices, API keys are strictly excluded from the source code. The scripts are configured to read your API keys directly from your system's environment variables.

Before running the experiments, you must set the corresponding environment variables in your terminal:

On Linux / macOS:

```bash
export GEMINI_API_KEY="your_actual_api_key_here"
export OPENAI_API_KEY="your_actual_api_key_here"
export ANTHROPIC_API_KEY="your_actual_api_key_here"
export GROQ_API_KEY="your_actual_api_key_here"
export HF_TOKEN="your_huggingface_token_here"
```
On Windows (Command Prompt):

```bash
set GEMINI_API_KEY="your_actual_api_key_here"
```

🔬 Usage

To run an evaluation, simply execute the desired Python script. The scripts will automatically generate the L∈{10,20,50,100} trajectories, interface with the respective APIs using Temperature=0.0, and output the deterministic evaluation into a newly generated .csv file.

```bash
python lab_openai.py
```

Note: The scripts include built-in rate-limiting safety measures (time.sleep() and exception handling for HTTP 429 errors) to respect the quota constraints of commercial API providers.
📊 Evaluation Metric

The scripts utilize a draconian binary evaluation metric. The regular expression module extracts the generated JSON from the <final_state> tags and compares it strictly against the mathematically calculated Ground Truth. Partial successes are recorded as 0 (Logical Collision), and only absolute topological matches are recorded as 1 (Absolute Success).
