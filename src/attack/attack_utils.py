import json
import os
import aiohttp
import asyncio
from tqdm import tqdm
from config.globals import working_dir, data_base_dir
import yaml
import multiprocessing
import time
from infact.tools.search.knowledge_base import KnowledgeBase

import re

# OpenAI API configuration
GPT_URL = "https://api.openai.com/v1/"

with open("config/api_keys.yaml", "r") as f:
    api_keys = yaml.load(f, Loader=yaml.FullLoader)
GPT_API_KEY = api_keys["openai_api_key"]
 
headers = {
    "Authorization": f"Bearer {GPT_API_KEY}",
    "Content-Type": "application/json",
}

def distribute_gpus(n_processes, gpu_ids):
    """
    Distribute GPUs to processes in a round-robin fashion.
    
    Args:
        n_processes: Number of processes
        gpu_ids: List of GPU IDs to distribute
        
    Returns:
        list: List of GPU assignments for each process
    """
    if not gpu_ids:
        return [None] * n_processes
    
    gpu_assignments = []
    for i in range(n_processes):
        gpu_assignments.append(gpu_ids[i % len(gpu_ids)])
    return gpu_assignments

def load_gt_data(gt_data_path: str):
    """
    Load ground truth data from JSON file.
    
    Args:
        gt_data_path: Path to the ground truth JSON file
        
    Returns:
        dict: Ground truth data
    """
    with open(gt_data_path, "r") as f:
        gt_data = json.load(f)
    return gt_data

def get_report_from_cache(cache_dir: str, claim_id: int):
    """
    Load fact-check report from cache directory.
    
    Args:
        cache_dir: Directory containing cached reports
        claim_id: ID of the claim
        
    Returns:
        str: Report content
    """
    report_path = os.path.join(cache_dir, f"{claim_id}")
    with open(report_path, "r") as f:
        report = f.read()
    return report

def get_log_from_cache(cache_dir: str, claim_id: int):
    """
    Load fact-check log from cache directory.
    
    Args:
        cache_dir: Directory containing cached logs
        claim_id: ID of the claim
        
    Returns:
        str: Log content
    """
    log_dir = cache_dir.replace("docs", "logs")
    report_path = os.path.join(log_dir, f"{claim_id}.txt")
    with open(report_path, "r") as f:
        logs = f.read()
    return logs

def get_query_from_log(log: str):
    """
    Extract questions and their corresponding search queries from a log.
    
    Args:
        log (str): The log content as a string
    
    Returns:
        dict: A dictionary where keys are questions and values are lists of search queries
    """
    result = {}
    current_question = None
    
    for line in log.split('\n'):
        line = line.strip()
        
        # Find questions
        if line.startswith("Answering question:"):
            current_question = line.split("Answering question:")[1].strip()
            result[current_question] = []
            
        # Find queries associated with current question
        elif current_question and "Searching averitec_kb with query:" in line:
            query = line.split("Searching averitec_kb with query:")[1].strip()
            result[current_question].append(query)
    return result

def get_all_valid_claim_ids(attack_set_dir, variant, logger):
    """
    Get all correctly predicted claim IDs with label support or refute.
    
    This function filters claims that:
    1. Have correct predictions (ground truth matches prediction)
    2. Have labels of "supported" or "refuted" (not "not_enough_info")
    
    Args:
        attack_set_dir: Directory containing attack dataset
        variant: Dataset variant (dev/train)
        logger: Logger instance
        
    Returns:
        list: List of valid claim IDs
    """
    # Load ground truth data
    gt_data_path = os.path.join(data_base_dir, "AVeriTeC", f"{variant}.json")
    try:
        gt_data = load_gt_data(gt_data_path)
    except Exception as e:
        logger.error(f"Failed to load gt data: {e}")
        return []
    
    valid_claim_ids = []
    supported_ids = []
    refuted_ids = []
    
    # Iterate through all files in cache directory
    for filename in tqdm(os.listdir(attack_set_dir), desc="Getting valid claims"):
        try:
            claim_id = int(filename)
            # Check if in gt_data and label is Supported or Refuted
            if gt_data[claim_id]['label'].lower() in ["supported", "refuted"]:
                # Get and parse report
                report = get_report_from_cache(attack_set_dir, claim_id)
                parsed_report = parse_report(report)

                # Check if prediction is correct
                gt_label = gt_data[claim_id]['label'].lower()
                pred_label = parsed_report['verdict'].lower()
                
                if gt_label == pred_label:
                    valid_claim_ids.append(claim_id)
                    if gt_label == "supported":
                        supported_ids.append(claim_id)
                    else:
                        refuted_ids.append(claim_id)
        except Exception as e:
            logger.warning(f"Error processing claim {filename}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Found {len(valid_claim_ids)} valid claims available for attack, including {len(supported_ids)} Supported claims and {len(refuted_ids)} Refuted claims")
    
    # Sort for consistent ordering
    valid_claim_ids.sort()
    return valid_claim_ids

def parse_report(report: str):
    """
    Parse the report into structured components.
    
    Args:
        report: Raw report string
        
    Returns:
        dict: Parsed report with claim, questions, answers, verdict, and justification
    """
    temp_dict = {
        "claim": "",
        "questions": [],
        "answers": [],
        "verdict": "",
        "justification": ""
    }
    
    # Extract claim
    if "## Claim" in report:
        claim_section = report.split("## Claim")[1].split("## Initial Q&A")[0].strip()
        temp_dict["claim"] = claim_section
    
    # Extract Q&A pairs
    if "## Initial Q&A" in report and "## Final Judgement" in report:
        qas = report.split("## Initial Q&A")[1].split("## Final Judgement")[0].strip()
        
        # Split by question (each starts with ###)
        qa_sections = qas.split("###")[1:]  # Skip the first empty part
        
        for qa_section in qa_sections:
            if "Answer:" in qa_section:
                parts = qa_section.split("Answer:", 1)
                question = parts[0].strip()
                answer = parts[1].strip()
                temp_dict["questions"].append(question)
                temp_dict["answers"].append(answer)
    
    # Extract verdict
    if "### Verdict:" in report:
        verdict_section = report.split("### Verdict:")[1].split("###", 1)[0].strip()
        temp_dict["verdict"] = verdict_section
    
    # Extract justification
    if "### Justification" in report:
        justification = report.split("### Justification")[1].strip()
        temp_dict["justification"] = justification
    
    return temp_dict
   
async def query_gpt_single(session: aiohttp.ClientSession, input: str, model_name: str="gpt-4o-mini", max_tokens: int=512, temperature: float=1):
    """
    Make a single API call to GPT model.
    
    Args:
        session: aiohttp session for API calls
        input: Input prompt
        model_name: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        str: Generated response or None if failed
    """
    try:
        async with session.post(
            url=f"{GPT_URL}chat/completions",
            json={
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": input}],
            },
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                return content
            else:
                print(f"Request failed with status code: {response.status}")
                return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

async def query_gpt_batch(input, max_limits: int, model_name: str="gemini-1.5-flash", max_tokens: int=512, temperature: float=1):
    """
    Make multiple API calls to GPT model in parallel.
    
    Args:
        input: Input prompt
        max_limits: Number of parallel requests
        model_name: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        list: List of generated responses
    """
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in tqdm(range(max_limits), desc="Creating tasks"):
            tasks.append(query_gpt_single(session, input, model_name, max_tokens, temperature))
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating Fake Evidence with API"):
            result = await task
            results.append(result)
        return results
 
def setup_process_logger(logs_dir, claim_id):
    """
    Set up separate logger for each process.
    
    Args:
        logs_dir: Directory to store log files
        claim_id: ID of the claim being processed
        
    Returns:
        logging.Logger: Configured logger instance
    """
    import logging

    # Create log format
    formatter = logging.Formatter(
        '%(asctime)s - Process(%(process)d) - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure file handler
    log_file = os.path.join(logs_dir, f"claim_{claim_id}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Create process-specific logger
    logger = logging.getLogger(f"process_{claim_id}")
    logger.setLevel(logging.DEBUG)

    # Ensure no duplicate handlers are added
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Prevent log propagation to root logger
    logger.propagate = False

    return logger

def setup_logging(logs_dir):
    """
    Set up logging configuration for the main process.
    
    Args:
        logs_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    import logging
    
    # Create log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure file handler
    log_file = os.path.join(logs_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Get root logger and add handlers
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing old handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logs will be saved to: {log_file}")
    
    return logger

def setup_experiment_dir(args):
    """
    Create and set up output directory structure for experiments.
    
    Args:
        args: Command line arguments containing experiment parameters
        
    Returns:
        dict: Dictionary containing paths to various experiment directories
    """
    # Create experiment name (use provided name or timestamp)
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{args.variant}_{args.attack_type}_{args.victim}_{args.poison_rate}"
    
    attack_cache_dir = os.path.join(working_dir, "attack", "attack_cache", args.variant)
    os.makedirs(attack_cache_dir, exist_ok=True)
    
    # Create main experiment directory
    exp_dir = os.path.join(args.out_dir, exp_name)
    print(f"exp_dir: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    logs_dir = os.path.join(exp_dir, "logs")
    results_dir = os.path.join(exp_dir, "results")
    resources_dir = os.path.join(exp_dir, "resources")
    knns_dir = os.path.join(exp_dir, "knns")
    
    for directory in [logs_dir, results_dir, resources_dir, knns_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Create configuration file containing experiment parameters
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Return directory structure
    return {
        "exp_dir": exp_dir,
        "logs_dir": logs_dir,
        "attack_cache_dir": attack_cache_dir,
        "results_dir": results_dir,
        "resources_dir": resources_dir,
        "knns_dir": knns_dir,
        "exp_name": exp_name
    }

class KBManager:
    """
    KB pool manager, responsible for creating and reusing KB instances.
    
    This class manages KnowledgeBase instances to avoid recreating them
    for each claim, which is expensive due to embedding model loading.
    """
    def __init__(self):
        self._kb_instances = {}
        self._lock = multiprocessing.Lock() if hasattr(multiprocessing, 'Lock') else None
    
    def get_kb(self, variant: str, device: str, logger) -> KnowledgeBase:
        """
        Get or create KB instance.
        
        Args:
            variant: Dataset variant
            device: Device to use (cpu/cuda)
            logger: Logger instance
            
        Returns:
            KnowledgeBase: KB instance
        """
        # In multi-process environment, each process has its own independent instance
        process_id = os.getpid()
        cache_key = f"{process_id}_{variant}_{device}"
        
        if cache_key not in self._kb_instances:
            logger.info(f"Creating new KB instance (variant={variant}, device={device})")
            start_time = time.time()
            kb = KnowledgeBase(variant=variant, device=device, logger=logger)
            end_time = time.time()
            logger.info(f" KB initialization took {end_time - start_time:.2f} seconds")
            self._kb_instances[cache_key] = kb
        else:
            logger.info(f"Reusing existing KB instance {cache_key}")
        
        return self._kb_instances[cache_key]
    
    def preload_kb(self, variant: str, device: str, logger):
        """
        Preload KB instance to avoid delay on first use.
        
        Args:
            variant: Dataset variant
            device: Device to use
            logger: Logger instance
        """
        logger.info(f"Preloading KB instance: variant={variant}, device={device}")
        self.get_kb(variant, device, logger)
    
    def clear_cache(self):
        """Clear cache of KB instances."""
        self._kb_instances.clear()
        
    def get_cache_info(self) -> dict:
        """
        Get cache information.
        
        Returns:
            dict: Information about cached instances
        """
        return {
            "cached_instances": len(self._kb_instances),
            "cache_keys": list(self._kb_instances.keys())
        }

def find_larger_cache(knns_dir: str, resources_dir: str, claim_id: int, current_num_fake: int, attack_type: str, logger, model_cache_suffix: str = ""):
    """
    Find if there exists a cache containing more fake evidence for sampling from.
    
    This function searches for existing cache files that contain more fake evidence
    than currently needed, allowing us to sample from them instead of regenerating.
    
    Args:
        knns_dir: Directory containing KNN model files
        resources_dir: Directory containing resource files
        claim_id: ID of the claim
        current_num_fake: Current number of fake evidence needed
        attack_type: Type of attack
        logger: Logger instance
        model_cache_suffix: Suffix for model-specific cache files
        
    Returns:
        tuple: (found, num_fake, knns_path, resources_path)
    """
    larger_cache_found = False
    larger_num_fake = 0
    larger_knns_path = ""
    larger_resources_path = ""
    
    # Search for matching pattern files in knns_dir
    try:
        # Build matching pattern suffix
        knns_suffix = f"_knns{model_cache_suffix}.pkl" if model_cache_suffix else "_knns.pkl"
        
        for filename in os.listdir(knns_dir):
            if filename.startswith(f"{claim_id}_") and filename.endswith(knns_suffix):
                # Parse filename: {claim_id}_{num_fake}_{attack_type}_knns{embed_suffix}{model_cache_suffix}.pkl
                # Remove suffix part to parse the front content
                base_name = filename.replace(knns_suffix, "")
                # Also need to remove embed_suffix (if exists)
                from config.globals import embedding_model
                embed_suffix = ""
                if embedding_model == "NovaSearch/stella_en_400M_v5":
                    embed_suffix = "_stella"
                elif embedding_model == "Qwen/Qwen3-Embedding-0.6B":
                    embed_suffix = "_qwen"
                elif embedding_model == "Snowflake/snowflake-arctic-embed-m-v2.0":
                    embed_suffix = "_snowflake"
                elif embedding_model == "Alibaba-NLP/gte-modernbert-base":
                    embed_suffix = "_modernbert"
                elif embedding_model == "Lajavaness/bilingual-embedding-base":
                    embed_suffix = "_bilingual"
                
                if embed_suffix and base_name.endswith(embed_suffix):
                    base_name = base_name[:-len(embed_suffix)]
                
                parts = base_name.split("_")
                if len(parts) >= 3:
                    try:
                        file_claim_id = int(parts[0])
                        file_num_fake = int(parts[1])
                        file_attack_type = "_".join(parts[2:])  # attack_type may contain underscores
                        
                        # Check if matches and contains more fake evidence
                        if (file_claim_id == claim_id and 
                            file_attack_type == attack_type and 
                            file_num_fake > current_num_fake):
                            
                            # Check if corresponding resources file exists
                            resources_filename = f"{claim_id}_{file_num_fake}_{attack_type}_resources{model_cache_suffix}.pkl"
                            resources_path = os.path.join(resources_dir, resources_filename)
                            
                            if os.path.exists(resources_path):
                                if file_num_fake > larger_num_fake:
                                    larger_cache_found = True
                                    larger_num_fake = file_num_fake
                                    larger_knns_path = os.path.join(knns_dir, filename) 
                                    larger_resources_path = resources_path
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Error finding larger cache: {e}")
    
    if larger_cache_found:
        logger.info(f"Found larger cache: {larger_num_fake} fake evidence, path: {larger_knns_path}")
    
    return larger_cache_found, larger_num_fake, larger_knns_path, larger_resources_path

