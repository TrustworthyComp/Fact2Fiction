

# Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-Checking Systems

This repository provides the official implementation of **Fact2Fiction**, the first poisoning attack framework designed to target agentic fact-checking systems.



## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (required for embedding models and local LLMs)
- API keys for OpenAI and HuggingFace

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TrustworthyComp/Fact2Fiction.git
   cd Fact2Fiction/src
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Step 1: Prepare Fact-Checking Results

To run attacks, first generate fact-checking results using the InFact/DEFAME system:

1. **Configure API Keys**
   ```bash
   python -m scripts.setup
   ```

2. **Optional: Configure Datasets & Paths**
   Edit `config/globals.py` to specify input/output directory paths. To evaluate on AVeriTeC, download the benchmark dataset [here](https://huggingface.co/chenxwh/AVeriTeC/tree/main/data).

3. **Build AVeriTeC Knowledge Base**
   ```bash
   # Generate embeddings for the knowledge base (may take time)
   python -m scripts.averitec.build
   ```

4. **Generate Fact-Checking Results**
   ```bash
   # Run fact-checking for DEFAME or InFact system
   # Use --procedure_variant summary for DEFAME, infact for InFact
   python -m scripts.averitec.evaluate --procedure_variant <variant>
   ```

### Step 2: Run Fact2Fiction Attacks

#### Basic Attack Example
Launch a Fact2Fiction attack against DEFAME with a 1% poisoning rate:

```bash
python -m attack.main \
    --attack-type fact2fiction \
    --victim defame \
    --poison-rate 0.01 \
    --gpu-ids 0 0 1 1 \
    --n-processes 4 \
    --fact-checker-model gpt-4o-mini \
    --attacker-model gpt-4o-mini
```

## Acknowledgments

This work use the codes from the [InFact/DEFAME fact-checking system](https://github.com/multimodal-ai-lab/DEFAME/tree/infact). We thank the authors for sharing their code.
