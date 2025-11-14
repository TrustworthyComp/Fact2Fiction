# AVeriTeC Experiment Scripts

This directory contains experiment scripts for running AVeriTeC benchmark tests, supporting different LLM models and procedure variants.

## Script Descriptions

### 1. `run_experiments.py` - Batch Run All Experiments

Automatically runs all 6 combinations (3 procedure_variants × 2 LLMs):

```bash
cd scripts/averitec
python run_experiments.py
```

**Experiment Combinations:**
- gemini-2.0-flash + defame
- gemini-2.0-flash + infact  
- gemini-2.0-flash + no_qa
- gpt-4o + defame
- gpt-4o + infact
- gpt-4o + no_qa

### 2. `run_single_experiment.py` - Run Single Experiment

Run a specified single experiment combination:

```bash
cd scripts/averitec
python run_single_experiment.py <llm> <procedure_variant>
```

**Examples:**
```bash
# Run Gemini 2.0 Flash + defame
python run_single_experiment.py gemini-2.0-flash defame

# Run GPT-4o + infact
python run_single_experiment.py gpt-4o infact

# Run Gemini 2.0 Flash + no_qa
python run_single_experiment.py gemini-2.0-flash no_qa
```

### 3. `evaluate.py` - Original Evaluation Script

The original evaluation script, currently configured for gpt_4o_mini + no_qa.

## Available Configurations

### LLM Models
- `gemini-2.0-flash`: Gemini 2.0 Flash
- `gpt-4o`: GPT-4o

### Procedure Variants
- `defame`: Maps to "summary" procedure (DynamicSummary)
- `infact`: Maps to "infact" procedure (InFact)
- `no_qa`: Maps to "no_qa" procedure (StaticSummary)

## Output Results

Experiment results will be saved in the `out/averitec/` directory, organized in the following structure:
```
out/averitec/
├── <procedure>/
│   └── <llm>/
│       └── <timestamp>/
│           ├── averitec_out.json
│           ├── logs/
│           └── ...
```

## Notes

1. All scripts use the same base configuration (temperature=0.01, max_iterations=3, etc.)
2. Uses multiprocessing, requires spawn start method to be set
3. Ensure sufficient API quota to run all experiments
4. The defame procedure actually uses the "summary" (DynamicSummary) implementation