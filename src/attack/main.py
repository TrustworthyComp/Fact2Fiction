"""
Fact2Fiction Attack Main Module

This module orchestrates the complete attack pipeline:
1. Loads fact-checking reports and extracts claims, verdicts, and justifications
2. Generates malicious evidence using various attack methods (Fact2Fiction, PoisonedRAG, etc.)
3. Injects malicious evidence into the knowledge base
4. Re-evaluates the fact-checking system to measure attack success
5. Computes attack metrics: Attack Success Rate (ASR), System Fail Rate (SFR), etc.

The attack implements a three-tier caching strategy:
- Tier 1: Exact match cache (same claim_id, num_fake, attack_type)
- Tier 2: Larger cache sampling (sample from cache with more fake evidence)
- Tier 3: Generate new fake evidence if no cache found

Reference: "Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System" (AAAI 2026)
"""

import os
import pickle
import numpy as np
import random
from tqdm import tqdm
from infact.common.modeling import make_model
from infact.fact_checker import FactChecker
from infact.tools.search.knowledge_base import KnowledgeBase
from infact.tools import initialize_tools
from attack.attack_methods import create_naive_attack, create_poisoned_rag_attack, create_prompt_injection_attack, create_fact2fiction_attack
from attack.attack_utils import get_report_from_cache, load_gt_data, parse_report, setup_process_logger, distribute_gpus, setup_experiment_dir, get_all_valid_claim_ids, KBManager, find_larger_cache
from sklearn.neighbors import NearestNeighbors
from config.globals import data_base_dir
import json
from infact.common.logger import Logger
from infact.utils.console import bold
import sys
import multiprocessing
import signal
import json_repair
import time
import torch
import asyncio

# Set random seed for reproducibility
random.seed(1126)
from config.globals import embedding_model

def get_model_cache_suffix(fact_checker_model: str, attacker_model: str, attack_type: str) -> str:
    """
    Generate cache suffix based on fact_checker_model and attacker_model.
    
    For fact2fiction related attacks (dependent on fact-checker output), need to distinguish fact_checker_model.
    For other attacks (such as poisoned_rag, disinformation, etc.), only need to distinguish attacker_model.
    
    Args:
        fact_checker_model: Model used for fact checking
        attacker_model: Model used for attack generation
        attack_type: Type of attack
        
    Returns:
        str: Cache suffix for model-specific files
    """
    # Normalize model names
    def normalize_model_name(model_name: str) -> str:
        model_name_map = {
            "gpt_4o_mini": "gpt-4o-mini",
            "gpt_4o": "gpt-4o",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
            "gpt-4o-2024-08-06": "gpt-4o",
            "gemini-2.0-flash": "gemini-2.0-flash",
        }
        return model_name_map.get(model_name, model_name)
    
    norm_fact_checker = normalize_model_name(fact_checker_model)
    norm_attacker = normalize_model_name(attacker_model)
    
    # fact2fiction dependent attack types
    justification_dependent_attacks = [
        "fact2fiction"
    ]
    
    if attack_type in justification_dependent_attacks:
        # For fact2fiction attacks, need to consider both models
        if norm_fact_checker == "gpt-4o-mini" and norm_attacker == "gpt-4o-mini":
            return ""
        else:
            return f"_fc-{norm_fact_checker}_att-{norm_attacker}"
    else:
        # For other attacks, only need to consider attacker model
        if norm_attacker == "gpt-4o-mini":
            return ""
        else:
            return f"_att-{norm_attacker}" 

# Set embedding model suffix based on configuration
if embedding_model == "Alibaba-NLP/gte-base-en-v1.5":
    embed_suffix = ""
elif embedding_model == "NovaSearch/stella_en_400M_v5":
    embed_suffix = "_stella"
elif embedding_model == "Qwen/Qwen3-Embedding-0.6B":
    embed_suffix = "_qwen"


# Attack type definitions
ATTACK_TYPES = ["naive", "disinformation", "poisoned_rag", "prompt_injection", "fact2fiction"]
VICTIM_SYSTEMS = ['infact', 'defame', 'no_qa']

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to standard Python types, ensuring JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        obj: Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Global KB manager instance
_kb_manager = KBManager()

def get_or_create_kb(variant: str, device: str, logger) -> KnowledgeBase:
    """
    Get or create KB instance, implementing reuse.
    
    Args:
        variant: Dataset variant
        device: Device to use
        logger: Logger instance
        
    Returns:
        KnowledgeBase: KB instance
    """
    logger.info(f"Getting or creating KB instance: variant={variant}, device={device}")
    temp_kb = _kb_manager.get_kb(variant, device, logger)
    temp_kb._setup_embedding_model()
    return temp_kb

def reset_kb_cache():
    """Reset KB cache, mainly for testing."""
    _kb_manager.clear_cache()

def add_fake_evidence_to_kb(kb: KnowledgeBase, claim_id: int, fake_evidences: list[dict], attack_type: str, exp_dirs: dict, logger, attacker_model: str, fact_checker_model: str):
    """
    Add fake evidence to specified claim and retrain its KNN model.
    
    Args:
        kb: KnowledgeBase instance
        claim_id: ID of the claim
        fake_evidences: List of fake evidence to add
        attack_type: Type of attack used
        exp_dirs: Experiment directory structure
        logger: Logger instance
        attacker_model: Model used for attack
        fact_checker_model: Model used for fact checking
        
    Returns:
        tuple: (updated_kb, embeddings)
    """
    # Ensure directories exist
    knns_dir = exp_dirs["knns_dir"]
    resources_dir = exp_dirs["resources_dir"]
    attack_cache_dir = exp_dirs["attack_cache_dir"]

    os.makedirs(knns_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)
    
    num_fake_evidences = len(fake_evidences)
    model_cache_suffix = get_model_cache_suffix(fact_checker_model, attacker_model, attack_type)
    temp_knns_path = os.path.join(knns_dir, f"{claim_id}_{num_fake_evidences}_{attack_type}_knns{embed_suffix}{model_cache_suffix}.pkl")
    temp_resources_path = os.path.join(resources_dir, f"{claim_id}_{num_fake_evidences}_{attack_type}_resources{model_cache_suffix}.pkl")
    
    # Clear cache to ensure we get new data next time
    kb.cached_resources = None
    kb.cached_resources_claim_id = None
    resources = kb._get_resources(claim_id)
    
    logger.info(f"Original evidence: {len(resources)} items, fake evidence: {len(fake_evidences)} items")

    # Load or generate original evidence embeddings
    embedding_cache_path = os.path.join(attack_cache_dir, f"{claim_id}_original_embeddings{embed_suffix}{model_cache_suffix}.pkl")
    if os.path.exists(embedding_cache_path):
        with open(embedding_cache_path, "rb") as f:
            original_embeddings = pickle.load(f)
        logger.info(f"Loaded original evidence embeddings cache from {embedding_cache_path}")
    else:
        original_embeddings = kb._embed_many([resource["url2text"] for resource in resources if resource["url2text"] is not None])
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(original_embeddings, f)
        logger.info(f"Saved original evidence embeddings cache to {embedding_cache_path}")

    # Maintain the same order as old version: original evidence first, fake evidence last
    all_resources = resources + fake_evidences
    logger.info(f"After adding fake evidence: {len(all_resources)} items")

    # Generate embeddings for fake evidence
    fake_embeddings = kb._embed_many([fake_evidence["url2text"] for fake_evidence in fake_evidences if fake_evidence["url2text"] is not None])
    
    # Merge embeddings, maintaining the same order as all_resources
    embeddings = np.concatenate([original_embeddings, fake_embeddings], axis=0)
    
    if len(embeddings) > 0:
        temp_knn = NearestNeighbors(n_neighbors=min(10, len(embeddings))).fit(embeddings)
    else:
        temp_knn = None
        
    # Save KNN model and fake evidence
    logger.info(f"Saving KNN model to {temp_knns_path}")
    with open(temp_knns_path, "wb") as f:
        pickle.dump(temp_knn, f)
        
    logger.info(f"Saving fake evidence to {temp_resources_path}")
    with open(temp_resources_path, "wb") as f:
        pickle.dump(fake_evidences, f)
    
    # Update kb object with new data
    kb.embedding_knns[claim_id] = temp_knn
    kb.cached_resources = all_resources  # Keep complete resources in memory for querying
    kb.cached_resources_claim_id = claim_id
    
    return kb, embeddings

def create_poisoned_evidence(kb: KnowledgeBase, parsed_fc_report: dict, poison_rate: float, all_fake_evidences: list[dict], attack_type: str, attacker_model: str, exp_dirs: dict, logger, fact_checker_model: str):
    """
    Create misleading fake evidence for target claim using the specified attack method.
    
    This function implements a three-tier caching strategy to optimize attack efficiency:
    1. Tier 1 (Exact Match): Check for cache with exact same parameters (claim_id, num_fake, attack_type)
    2. Tier 2 (Larger Cache Sampling): Look for cache with more fake evidence and sample from it
    3. Tier 3 (Generation): Generate new fake evidence if no suitable cache found
    
    The poisoning rate can be specified as:
    - A proportion (0.01 = 1%): num_fake = ⌈num_original · poison_rate / (1 - poison_rate)⌉
    - An absolute count (≥ 1.0): num_fake = poison_rate
    
    Args:
        kb: KnowledgeBase instance containing original evidence
        parsed_fc_report: Parsed fact-checking report with claim, verdict, and justification
        poison_rate: Poisoning rate (proportion 0-1 or absolute count ≥ 1.0)
        all_fake_evidences: Existing fake evidence already injected (for incremental attacks)
        attack_type: Type of attack ("naive", "disinformation", "poisoned_rag", "prompt_injection", "fact2fiction")
        attacker_model: LLM model used for attack generation
        exp_dirs: Experiment directory structure (knns_dir, resources_dir, etc.)
        logger: Logger instance for logging
        fact_checker_model: LLM model used by the victim fact-checking system
        
    Returns:
        tuple: (fake_evidences, embeddings) where:
            - fake_evidences: List of fake evidence dictionaries with url and url2text
            - embeddings: None (embeddings are generated later in add_fake_evidence_to_kb)
    """
    claim_id = parsed_fc_report["claim_id"]
    knns_dir = exp_dirs["knns_dir"]
    resources_dir = exp_dirs["resources_dir"]

    # Map model names to full versions
    model_name_map = {
        "gpt_4o_mini": "gpt-4o-mini-2024-07-18",
        "gpt_4o": "gpt-4o-2024-08-06",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o-2024-08-06",
    }
    if attacker_model in model_name_map:
        attacker_model_full = model_name_map[attacker_model]
    else:
        attacker_model_full = attacker_model
        
    original_resources = kb._get_resources(claim_id)
    if not original_resources:
        logger.warning(f"Warning: claim {claim_id} has no original evidence")
        return []
        
    urls = list(set([resource.get("url", "") for resource in original_resources if resource.get("url")]))
    if not urls:
        logger.warning(f"Warning: claim {claim_id} has no valid URLs")
        urls = [f"https://example.com/evidence/{i}" for i in range(5)]
    
    # Calculate number of fake evidence needed
    if len(all_fake_evidences) == 0:
        num_original = len(original_resources)
    else:
        num_original = len(original_resources) - len(all_fake_evidences)
        
    if poison_rate < 1.0:
        num_fake = max(1, int(num_original * poison_rate / (1 - poison_rate)))
    else:
        logger.info(f"Using {poison_rate} fake evidence count")
        num_fake = int(poison_rate)
        
    all_fake = int(num_fake) + len(all_fake_evidences)

    # Ensure directories exist
    os.makedirs(knns_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    model_cache_suffix = get_model_cache_suffix(fact_checker_model, attacker_model, attack_type)
    temp_knns_path = os.path.join(knns_dir, f"{claim_id}_{all_fake}_{attack_type}_knns{embed_suffix}{model_cache_suffix}.pkl")
    temp_resources_path = os.path.join(resources_dir, f"{claim_id}_{all_fake}_{attack_type}_resources{model_cache_suffix}.pkl")
    
    logger.info(f"Target KNN model path: {temp_knns_path}")
    logger.info(f"Target resources path: {temp_resources_path}")
    
    knn_exists = os.path.exists(temp_knns_path)
    resources_exists = os.path.exists(temp_resources_path)
    
    logger.info(f"Exact match cache - KNN model exists: {knn_exists}, resources exist: {resources_exists}")
    
    # ===== TIER 1: Exact Match Cache =====
    # Check if we have cached results with exact same parameters
    # This avoids regenerating fake evidence for the same attack configuration
    if knn_exists and resources_exists:
        logger.info(f"Exact match KNN model and {num_fake} fake evidence already exist for claim {claim_id}, loading directly")
        try:
            # Load KNN model
            with open(temp_knns_path, "rb") as f:
                kb.embedding_knns[claim_id] = pickle.load(f)
            
            # Load fake evidence
            fake_evidences = pickle.load(open(temp_resources_path, "rb"))
            
            # Get original resources
            kb.cached_resources = None
            kb.cached_resources_claim_id = None
            original_resources = kb._get_resources(claim_id)
            
            # Merge original resources and fake evidence
            all_resources = original_resources + fake_evidences
            
            # Update knowledge base cache
            kb.cached_resources = all_resources
            kb.cached_resources_claim_id = claim_id
            
            logger.info(f"Loaded {len(original_resources)} original evidence and {len(fake_evidences)} fake evidence")
            return kb, None
        except Exception as e:
            logger.error(f"Failed to load exact match cache: {e}, trying to find larger cache")
    
    # ===== TIER 2: Larger Cache Sampling =====
    # If exact match not found, look for a cache with more fake evidence
    # We can sample the required quantity from the larger cache, avoiding regeneration
    larger_cache_found, larger_num_fake, larger_knns_path, larger_resources_path = find_larger_cache(
        knns_dir, resources_dir, claim_id, num_fake, attack_type, logger, model_cache_suffix)
    
    if larger_cache_found:
        logger.info(f"Found larger cache: {larger_num_fake} fake evidence, will sample {num_fake} from it")
        try:
            # Load fake evidence from larger cache
            larger_fake_evidences = pickle.load(open(larger_resources_path, "rb"))
            logger.info(f"Loaded {len(larger_fake_evidences)} fake evidence from larger cache")
            
            # Sample required quantity from larger cache
            if len(larger_fake_evidences) >= num_fake:
                sampled_fake_evidences = random.sample(larger_fake_evidences, num_fake)
                logger.info(f"Sampled {len(sampled_fake_evidences)} from {len(larger_fake_evidences)} fake evidence")
                return sampled_fake_evidences, None
            else:
                logger.warning(f"Fake evidence count in larger cache ({len(larger_fake_evidences)}) is less than required count ({num_fake}), will regenerate fake evidence")
        except Exception as e:
            logger.error(f"Failed to sample from larger cache: {e}, will regenerate fake evidence")
    
    # ===== TIER 3: Generate New Fake Evidence =====
    # No suitable cache found, generate new fake evidence using the specified attack method
    logger.info(f"No suitable cache found, need to recreate {num_fake} {attack_type} type fake evidence")
    fake_evidences = []
    
    # Generate fake evidence based on attack type
    # Each attack method implements different strategies for crafting malicious content
    if attack_type == "naive":
        results = create_naive_attack(parsed_fc_report, num_fake)
    elif attack_type == "prompt_injection":
        results = create_prompt_injection_attack(parsed_fc_report, num_fake)
    elif attack_type == "disinformation":
        results = asyncio.run(create_poisoned_rag_attack(parsed_fc_report, attacker_model_full, num_fake, concat_query=False))
    elif attack_type == "poisoned_rag":
        results = asyncio.run(create_poisoned_rag_attack(parsed_fc_report, attacker_model_full, num_fake, concat_query=True))
    elif attack_type == "fact2fiction":
        results = asyncio.run(create_fact2fiction_attack(parsed_fc_report, attacker_model_full, num_fake, concat_query=True, weighted=True, use_justification=True))
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
        
    random.shuffle(results)
    for result in results:
        random_url = random.choice(list(urls))
        random_url = random_url + "/created"  # Mark as fake evidence
        fake_evidences.append({"url": random_url, "url2text": result["created_evidence"]})
    return fake_evidences, None

def attack_claim_wrapper(claim_id, poison_rate, attack_type, victim, variant, device, 
                        attack_set_dir, exp_dirs, attacker_model, fact_checker_model,
                        out_file, lock, gpu_id=None):
    """Wrap attack function, handle result writing and exceptions"""
    # Set GPU for current process
    if gpu_id is not None:
        proc_device = f"cuda:{gpu_id}" 
    else:
        proc_device = device
        
    # Create process-level logger
    proc_logger = setup_process_logger(exp_dirs["logs_dir"], claim_id)
    proc_logger.info(f"Process started processing claim {claim_id}, using GPU: {gpu_id if gpu_id is not None else 'None'}")
    
    try:
        result = attack_single_claim(
            claim_id=claim_id, 
            poison_rate=poison_rate, 
            attack_type=attack_type, 
            victim=victim,
            variant=variant, 
            device=proc_device,  # Use process-specific device settings
            attack_set_dir=attack_set_dir, 
            exp_dirs=exp_dirs, 
            logger=proc_logger,
            attacker_model=attacker_model,
            fact_checker_model=fact_checker_model
        )
        
        # Safely write results using lock
        with lock:
            with open(out_file, "a+") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            # sort the result
            with open(out_file, "r") as f:
                results = [json.loads(line) for line in f]
            results.sort(key=lambda x: int(x["claim_id"]))
            with open(out_file, "w") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            # Create summary report and print updated results
            create_summary_report(out_file, exp_dirs["results_dir"], fact_checker_model, attacker_model, proc_logger)

        proc_logger.info(f"Successfully completed attack for claim {claim_id}")
        return True
    except Exception as e:
        proc_logger.error(f"Error attacking claim {claim_id}: {e}")
        import traceback
        proc_logger.error(traceback.format_exc())
        return False

# Move attack_task function to global scope
def attack_task(args_tuple):
    """Function to process a single task, provided to process pool"""
    claim_id, kwargs = args_tuple
    return attack_claim_wrapper(claim_id, **kwargs)

def attack_all_claims(args, exp_dirs, logger):
    """Attack all valid claims and calculate success rate"""
    # Extract parameters
    attack_set_dir = args.attack_set_dir
    variant = args.variant
    poison_rate = args.poison_rate
    attack_type = args.attack_type
    victim = args.victim
    limit = args.limit
    attacker_model = args.attacker_model
    fact_checker_model = args.fact_checker_model
    n_processes = args.n_processes
    gpu_ids = args.gpu_ids

    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attack_target_dir = os.path.join(attack_set_dir, victim, fact_checker_model, "search_top_five", "docs")
    valid_claim_ids = get_all_valid_claim_ids(attack_target_dir, variant, logger)

    logger.info(f"number of valid_claim_ids: {len(valid_claim_ids)}")
    if not valid_claim_ids:
        logger.error("No valid claims found, cannot perform attack")
        return 0
    
    if limit is not None and limit > 0:
        logger.info(f"Limiting to {limit} claims for testing")
        valid_claim_ids = valid_claim_ids[:limit]
    
    fact_checker_suffix = f"_{fact_checker_model}" if fact_checker_model != "gpt-4o-mini" else ""
    attacker_suffix = f"_att_{attacker_model}" if attacker_model != "gpt-4o-mini" else ""
    results_dir = exp_dirs["results_dir"]
    logger.info(f"embedding_model: {embedding_model}")
    logger.info(f"embed_suffix: {embed_suffix}")
    logger.info(f"fact_checker_model: {fact_checker_model}")
    logger.info(f"fact_checker_suffix: {fact_checker_suffix}")
    logger.info(f"attacker_model: {attacker_model}")
    logger.info(f"attacker_suffix: {attacker_suffix}")
    out_file = os.path.join(results_dir, f"attack_results{embed_suffix}{fact_checker_suffix}{attacker_suffix}.jsonl")
    
    # Check processed claims
    processed_claim_ids = set()
    if os.path.exists(out_file):
        logger.info(f"Output file {out_file} already exists, loading processed results")
        try:
            with open(out_file, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        processed_claim_ids.add(int(result["claim_id"]))
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error reading processed results: {e}")
    
    # Filter out processed claims
    valid_claim_ids = [claim_id for claim_id in valid_claim_ids if claim_id not in processed_claim_ids]
    logger.info(f"Processed {len(processed_claim_ids)} claims, {len(valid_claim_ids)} claims remaining")

    if not valid_claim_ids:
        logger.info("All claims processed, generating summary report")
        create_summary_report(out_file, results_dir, fact_checker_model, attacker_model, logger)
        return 1
    
    # Assign GPUs to each process
    if gpu_ids:
        logger.info(f"Using GPU IDs: {gpu_ids}")
        gpu_assignments = distribute_gpus(n_processes, gpu_ids)
        logger.info(f"Process GPU assignments: {gpu_assignments}")
    else:
        gpu_assignments = [None] * n_processes
    
    # Create a lock for multi-process writing
    manager = multiprocessing.Manager()
    try:
        lock = manager.Lock()
        
        # Create task list
        tasks = []
        for i, claim_id in enumerate(valid_claim_ids):
            # Select GPU based on process index
            proc_idx = i % n_processes
            gpu_id = gpu_assignments[proc_idx]
            
            # Create task
            task = (
                claim_id,  # First parameter is claim_id
                {
                    "poison_rate": poison_rate,
                    "attack_type": attack_type,
                    "victim": victim,
                    "variant": variant,
                    "device": device,
                    "attack_set_dir": attack_set_dir,
                    "exp_dirs": exp_dirs,
                    "attacker_model": attacker_model,
                    "fact_checker_model": fact_checker_model,
                    "out_file": out_file,
                    "lock": lock,
                    "gpu_id": gpu_id,

                }
            )
            tasks.append(task)
        
        # Create process pool
        logger.info(f"Using {n_processes} processes to parallelize attacks")
        
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Parallel execution of attacks, using global attack_task function
            results = list(tqdm(
                pool.imap_unordered(attack_task, tasks),
                total=len(tasks),
                desc="Parallelizing attacks"
            ))
        
        # Count results
        success_count = sum(1 for r in results if r)
        fail_count = len(valid_claim_ids) - success_count
        logger.info(f"Attack completed: Success {success_count}, Fail {fail_count}")
        
    finally:
        # Ensure Manager is closed correctly
        try:
            manager.shutdown()
        except Exception as e:
            logger.warning(f"Warning closing Manager: {e}")
    
    # Generate summary report
    try:
        create_summary_report(out_file, results_dir, fact_checker_model, attacker_model, logger)
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
    
    return 1

def cleanup_resources():
    """Clean up process resources"""
    # Remove problematic resource cleanup code, let Python handle it automatically
    pass

def terminate_all_processes():
    """Terminate all processes"""
    try:
        # Send SIGTERM signal to parent process
        os.kill(os.getppid(), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        # Send SIGTERM signal to current process group
        os.killpg(os.getpgid(0), signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Force exit current process
    sys.exit(1)

def attack_single_claim(claim_id: int, poison_rate: float, attack_type: str, victim: str, variant: str, device: str, 
                attack_set_dir: str, exp_dirs: dict, logger, attacker_model=None, fact_checker_model=None):
    """
    Perform poisoning attack on a single claim and measure attack effectiveness.
    
    This function implements the complete attack pipeline:
    1. Load original fact-checking report and extract claim, verdict, justification
    2. Generate malicious evidence using the specified attack method
    3. Inject fake evidence into the knowledge base
    4. Re-run fact-checking with poisoned knowledge base
    5. Measure attack success: ASR (verdict inversion) and SFR (any incorrect verdict)
    6. Track evidence usage statistics (retrieval rates, fake evidence usage)
    
    Attack Success Rate (ASR): Proportion of claims where verdict is inverted (Supported ↔ Refuted)
    System Fail Rate (SFR): Proportion of claims with any incorrect verdict (broader than ASR)
    
    Args:
        claim_id: ID of the claim to attack
        poison_rate: Poisoning rate (proportion or absolute count)
        attack_type: Type of attack method to use
        victim: Victim system type ("defame", "infact", "no_qa")
        variant: Dataset variant ("dev" or "train")
        device: Device to use ("cuda" or "cpu")
        attack_set_dir: Directory containing original fact-checking results
        exp_dirs: Experiment directory structure
        logger: Logger instance
        attacker_model: LLM model for attack generation
        fact_checker_model: LLM model for victim fact-checking system
        
    Returns:
        dict: Attack results containing claim_id, attack_success, fact_check_fail,
              evidence usage statistics, and other metrics
    """
    logger.info(f"Starting attack for claim {claim_id}, using attacker model: {attacker_model}, fact checker model: {fact_checker_model}")
    try:
        # Reuse KB instance
        kb = get_or_create_kb(variant, device, logger)
        
        # Load gt data
        gt_data_path = os.path.join(data_base_dir, "AVeriTeC", f"{variant}.json")
        gt_data = load_gt_data(gt_data_path)
        gt_claim_data = gt_data[claim_id]
        del gt_data

        defame_dir = os.path.join(attack_set_dir, "defame", fact_checker_model, "search_top_five", "docs")
        infact_dir = os.path.join(attack_set_dir, "infact", fact_checker_model, "search_top_five", "docs")
        target_dir = defame_dir if variant == "defame" else infact_dir
        original_report = get_report_from_cache(target_dir, claim_id)
        parsed_fc_report = parse_report(original_report)
        parsed_fc_report["claim_id"] = claim_id
        original_justification = parsed_fc_report["justification"]
        original_verdict = parsed_fc_report["verdict"]
        parsed_fc_report["original_justification"] = original_justification
        parsed_fc_report["original_verdict"] = original_verdict
        ground_truth_label = gt_claim_data['label'].lower()
        
        # Initialize attack results
        attack_success = False
        fact_check_fail = False
        all_fake_evidences = []


        
        # Iterate attacks, up to 3 rounds (currently set to 1 for single-round attacks)
        # Multi-round attacks could adaptively refine based on previous round results
        max_rounds = 1 
        
        clean_kb = kb
        original_resources = clean_kb._get_resources(claim_id)
 
        pred_label = None
        
        for round_idx in range(max_rounds):
            if attack_success:
                logger.info(f"Attack successful in round {round_idx + 1}, stopping subsequent attacks")
                break
                
            logger.info(f"Starting round {round_idx + 1}...")
            
            # Create fake evidence based on current justification
            # For Fact2Fiction, this enables adaptive attacks that exploit updated justifications
            if round_idx == 0:
                # First round: Use original fact-checking report with initial justification
                # This is the standard attack scenario where we exploit the pre-attack justification
                attack_report = parsed_fc_report.copy()
                # Keep the initial justification and verdict
            else:
                # Subsequent rounds: Update report with results from previous round
                # This enables adaptive attacks that refine strategy based on intermediate results
                attack_report = parsed_fc_report.copy()
                attack_report["justification"] = current_justification
                attack_report["verdict"] = pred_label.value if pred_label else original_verdict
            
            # Create fake evidence
            
            logger.info(f"Round {round_idx + 1}: Creating poisoned evidence for claim {claim_id}, poison rate: {poison_rate}...")
            logger.info(f"Round {round_idx + 1}: Current accumulated fake evidence count: {len(all_fake_evidences)}")
            
            new_fake_evidences, all_embeddings = create_poisoned_evidence(
                clean_kb, attack_report, poison_rate, all_fake_evidences, attack_type, attacker_model, exp_dirs, logger, fact_checker_model
            )
            if all_embeddings is not None:
                all_embeddings = all_embeddings
            else:
                all_embeddings = []
            # Add new fake evidence to the accumulated evidence list
            if isinstance(new_fake_evidences, list):
                all_fake_evidences.extend(new_fake_evidences)
                logger.info(f"Round {round_idx + 1}: Accumulated fake evidence count: {len(all_fake_evidences)}")
                # Add new fake evidence to the original clean knowledge base
                current_kb, embeddings = add_fake_evidence_to_kb(clean_kb, claim_id, all_fake_evidences, attack_type, exp_dirs, logger, attacker_model, fact_checker_model)
                all_embeddings = embeddings
            elif isinstance(new_fake_evidences, KnowledgeBase):
                # If create_poisoned_evidence returns a KnowledgeBase object, use it directly
                logger.info(f"Round {round_idx + 1}: Using pre-poisoned knowledge base...")
                current_kb = new_fake_evidences
                poisoned_resources = current_kb._get_resources(claim_id)
                logger.info(f"Original resources in clean knowledge base: {len(original_resources)}")
                logger.info(f"Resources in poisoned knowledge base: {len(poisoned_resources)}")
                logger.info(f"Difference in resources between poisoned and original: {len(poisoned_resources) - len(original_resources)}")
                # Calculate new fake evidence
                round_fake_evidences = [
                    {"url": poisoned_resource["url"], "url2text": poisoned_resource["url2text"]} 
                    for poisoned_resource in poisoned_resources 
                    if poisoned_resource not in original_resources]
                assert len(round_fake_evidences) == len(poisoned_resources) - len(original_resources), f"Number of new fake evidence does not match difference in poisoned resources and original resources {len(round_fake_evidences)} != {len(poisoned_resources) - len(original_resources)}"
                all_fake_evidences.extend(round_fake_evidences)
                logger.info(f"Round {round_idx + 1}: Created {len(round_fake_evidences)} new fake evidence items from cached resources")
            # ===== Re-evaluate Fact-Checking System with Poisoned Knowledge Base =====
            # Initialize the fact-checking system (victim) with the poisoned knowledge base
            # This measures whether the attack successfully misleads the system
            
            # Configure LLM for fact-checking
            llm_kwargs = dict(
                temperature=0.01,  # Low temperature for deterministic fact-checking
                device=device,
            )
            llm = make_model(fact_checker_model, logger=logger, **llm_kwargs)

            # Map victim system names to procedure variants
            # DEFAME uses dynamic summary-based procedure, InFact uses explicit QA-based procedure
            victim_map = {
                "defame": "summary",  # DEFAME: implicit dynamic claim decomposition
                "infact": "infact",   # InFact: explicit sub-question decomposition
                "no_qa": "no_qa",     # Simple: naive RAG without decomposition
            }
            
            # Configure fact-checker with victim-specific parameters
            fact_checker_kwargs = dict(
                procedure_variant=victim_map[victim],  # Procedure type for the victim system
                max_iterations=3 if victim == "defame" else 1,  # DEFAME allows multiple iterations
                max_result_len=64_000,  # Maximum result length in characters
            )
            tools_config=dict(
                searcher=dict(
                    search_engine_config=dict(
                        averitec_kb=dict(variant=variant),
                    ),
                    limit_per_search=5,

                )
            )    
            # Initialize search tools and replace the knowledge base with poisoned version
            tools = initialize_tools(tools_config, logger=logger)
            searcher = tools[0]
            original_averitec_kb = searcher.search_apis["averitec_kb"]  # Original clean KB
            searcher.search_apis["averitec_kb"] = current_kb  # Replace with poisoned KB
            assert searcher.search_apis["averitec_kb"] is not original_averitec_kb, f"averitec_kb instance not successfully changed"
            
            # Set claim_id for both KBs to ensure proper resource retrieval
            original_averitec_kb.current_claim_id = claim_id
            current_kb.current_claim_id = claim_id

            # Verify that poisoning was successful (poisoned KB should have more resources)
            assert len(original_averitec_kb._get_resources()) != len(current_kb._get_resources()), f"Resources not poisoned, length different {len(original_averitec_kb._get_resources())} != {len(current_kb._get_resources())}"

            

            
            # Execute fact-checking with poisoned knowledge base
            # This measures whether the injected malicious evidence successfully misleads the system
            fc = FactChecker(
                llm=llm,
                tools=tools,
                logger=logger,
                **fact_checker_kwargs,
            )    
            logger.info("-"*100)
            logger.info(f"Round {round_idx + 1}: Executing fact check with poisoned knowledge base...")
            report, meta = fc.verify_claim(parsed_fc_report["claim"])
            pred_label = report.verdict
            current_justification = report.justification
            
            # ===== Measure Attack Effectiveness =====
            # Compare predicted verdict with ground truth to compute attack metrics
            logger.info(f"Round {round_idx + 1}: Predicted label: {pred_label.value}, Ground truth label: {ground_truth_label}")
            
            # Check if attack was successful
            # Attack Success (ASR): Verdict is inverted (Supported ↔ Refuted)
            # System Fail (SFR): Any incorrect verdict (broader than ASR)
            if pred_label.value != ground_truth_label:
                fact_check_fail = True  # System Fail Rate: any incorrect verdict
                if ((pred_label.value == "supported" and ground_truth_label == "refuted") or
                    (pred_label.value == "refuted" and ground_truth_label == "supported")):
                    attack_success = True  # Attack Success Rate: exact verdict inversion
                    logger.info(f"Round {round_idx + 1}: {bold(f'Attack successful')}")
                else:
                    logger.info(f"Round {round_idx + 1}: {bold(f'Fact check failed, but did not achieve the expected opposite conclusion')}")
            else:
                logger.info(f"Round {round_idx + 1}: {bold(f'Attack failed, fact check result correct')}")
                
        # ===== Evidence Usage Statistics =====
        # Track which evidence was actually retrieved and used by the fact-checking system
        # This measures the retrievability of malicious evidence (Successful Injection Rate, SIR)
        used_evidences = meta['used_evidence']
        used_evidence_set = set()
        
        # Extract evidence sources (URLs) from used evidence
        # DEFAME uses nested results structure, while InFact uses flat structure
        if victim != "defame":
            for evidence in used_evidences:
                used_evidence_set.add(evidence.source)
        else:
            for evidence in used_evidences:
                results = evidence.results
                for result in results:
                    used_evidence_set.add(result.source)

        used_evidences = list(used_evidence_set)
        # Identify fake evidence by checking for "/created" marker in URLs
        fake_evidence = [evidence for evidence in used_evidences if 'created' in evidence]
        original_evidence = [evidence for evidence in used_evidences if 'created' not in evidence]
        num_original_urls = len(original_evidence)
        num_fake_urls = len(fake_evidence)
        num_used_evidence_urls = len(used_evidences)
        
        # Successful Injection Rate (SIR): proportion of retrieved malicious evidence
        fake_evidence_usage_rate = num_fake_urls / num_used_evidence_urls if num_used_evidence_urls > 0 else 0
        
        # Calculate fake evidence proportion (used fake / total fake generated)
        # For prompt injection, this is calculated differently since it modifies existing evidence
        if attack_type == "prompt_injection":
            # Prompt injection modifies existing evidence rather than adding new evidence
            # Denominator should be the number of evidence modified, not all_fake_evidences
            valid_resources = [r for r in original_resources if r.get("url2text") and r["url2text"].strip()]
            num_injected = max(1, int(len(valid_resources) * poison_rate)) if poison_rate < 1.0 else min(int(poison_rate), len(valid_resources))
            fake_evidence_proportion = num_fake_urls / num_injected if num_injected > 0 else 0
        else:
            # For other attacks: proportion of generated fake evidence that was actually used
            fake_evidence_proportion = num_fake_urls / len(all_fake_evidences) if len(all_fake_evidences) > 0 else 0

        logger.info(f"  - =================================================")
        logger.info(f"Evidence usage statistics:")
        logger.info(f"  - Retrieval evidence usage statistics:")
        logger.info(f"  - Number of original evidence used: {num_original_urls}")
        logger.info(f"  - Number of fake evidence used: {num_fake_urls}")
        logger.info(f"  - Total number of evidence used: {num_used_evidence_urls}")
        logger.info(f"  - Fake evidence usage rate (fake evidence used / total fake evidence): {fake_evidence_usage_rate:.2%}")
        logger.info(f"  - Fake evidence proportion (fake evidence used / total evidence used): {fake_evidence_proportion:.2%}")
        # Return attack results
        result = {
            "claim_id": int(claim_id),
            "claim": parsed_fc_report["claim"],
            "variant": variant,
            "attack_type": attack_type,      
            "poison_rate": float(poison_rate),
            "fact_checker_model": fact_checker_model,
            "attacker_model": attacker_model,
            "attack_success": bool(attack_success),
            "fact_check_fail": bool(fact_check_fail),
            "gt_label": ground_truth_label,
            "pred_label": pred_label.value if pred_label else None,
            "before_justification": original_justification,
            "after_justification": current_justification,
            "rounds_needed": int(round_idx + 1 if 'round_idx' in locals() else 0),
            "total_fake_evidence": int(len(all_fake_evidences)),
            # New evidence usage statistics
            "used_original_evidence": int(num_original_urls),
            "used_fake_evidence": int(num_fake_urls),
            "total_used_evidence": int(num_used_evidence_urls),
            "fake_evidence_usage_rate": float(fake_evidence_usage_rate),
            "fake_evidence_proportion": float(fake_evidence_proportion),

        }
        
        # Ensure all values are JSON serializable
        return convert_numpy_types(result)
    except Exception as e:
        logger.error(f"Error attacking claim {claim_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Simply re-raise the exception, let the upper layer handle it
        raise

# Modify create_summary_report to return summary data
def create_summary_report(results_file, results_dir, fact_checker_model, attacker_model, logger):
    """Generate summary report based on results file and return summary data"""
    try:
        results = []
        with open(results_file, "r") as f:
            for line in f:
                try:
                    result = json_repair.loads(line)
                    results.append(result)
                except:
                    continue
        
        if not results:
            logger.warning(f"No result data found, cannot generate summary report. Results file: {results_file}")
            return None
        
        # Calculate statistics
        success_count = sum(1 for r in results if r.get("attack_success", False))
        fail_count = sum(1 for r in results if r.get("fact_check_fail", False))
        
        # Evidence usage statistics - Retrieval stage
        total_used_original_evidence = sum(r.get("used_original_evidence", 0) for r in results)
        total_used_fake_evidence = sum(r.get("used_fake_evidence", 0) for r in results)
        total_used_evidence = sum(r.get("total_used_evidence", 0) for r in results)
        
        # Calculate averages - Retrieval stage
        avg_used_original_evidence = total_used_original_evidence / len(results) if results else 0
        avg_used_fake_evidence = total_used_fake_evidence / len(results) if results else 0
        avg_total_used_evidence = total_used_evidence / len(results) if results else 0
        avg_fake_evidence_usage_rate = total_used_fake_evidence / total_used_evidence if total_used_evidence > 0 else 0
        avg_fake_evidence_proportion = sum(r.get("fake_evidence_proportion", 0) for r in results) / len(results) if results else 0
        

        summary = {
            "total_claims": len(results),
            "attack_success_rate": success_count / len(results) if results else 0,
            "fact_check_fail_rate": fail_count / len(results) if results else 0,
            
            # Retrieval evidence usage statistics
            "avg_used_original_evidence": avg_used_original_evidence,
            "avg_used_fake_evidence": avg_used_fake_evidence,
            "avg_total_used_evidence": avg_total_used_evidence,
            "avg_fake_evidence_usage_rate": avg_fake_evidence_usage_rate,
            "avg_fake_evidence_proportion": avg_fake_evidence_proportion,
            

            
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": {
                "attack_success_count": success_count,
                "fact_check_fail_count": fail_count,
                
                # Detailed statistics for Retrieval stage
                "total_used_original_evidence": total_used_original_evidence,
                "total_used_fake_evidence": total_used_fake_evidence,
                "total_used_evidence": total_used_evidence,

            }
        }
        
        fact_checker_suffix = f"_{fact_checker_model}" if fact_checker_model != "gpt-4o-mini" else ""
        attacker_suffix = f"_att_{attacker_model}" if attacker_model != "gpt-4o-mini" else ""
        # Save summary report
        summary_file = os.path.join(results_dir, f"summary{embed_suffix}{fact_checker_suffix}{attacker_suffix}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_file}")
        logger.info(f"Number of claims processed: {len(results)}")
        logger.info(f"Attack success rate: {summary['attack_success_rate']:.2%}")
        logger.info(f"Fact check fail rate: {summary['fact_check_fail_rate']:.2%}")
        logger.info(f"Average fake evidence usage rate: {summary['avg_fake_evidence_usage_rate']:.2%}")
        
        # Log Retrieval stage evidence usage statistics
        logger.info("Retrieval stage evidence usage statistics:")
        logger.info(f"  - Average number of original evidence used: {summary['avg_used_original_evidence']:.2f}")
        logger.info(f"  - Average number of fake evidence used: {summary['avg_used_fake_evidence']:.2f}")
        logger.info(f"  - Average total number of evidence used: {summary['avg_total_used_evidence']:.2f}")
        logger.info(f"  - Average fake evidence usage rate*: {summary['avg_fake_evidence_usage_rate']:.2%}")
        logger.info(f"  - Average fake evidence proportion: {summary['avg_fake_evidence_proportion']:.2%}")
        

        
        # Return summary data for caller
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return None

def parse_args():
    """Parse command line arguments"""
    import argparse
    from config.globals import working_dir
    parser = argparse.ArgumentParser(description="Knowledge Base Poisoning Attack Experiment")
    parser.add_argument("--attack_set_dir", type=str, 
                        default=os.path.join(working_dir, "out", "averitec"),
                        help="Attack dataset directory path")
    parser.add_argument("--out-dir", type=str, 
                        default=os.path.join(working_dir, "attack", "attack_results"),
                        help="Root directory for experiment outputs")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name, used to create subdirectories, defaults to timestamp")
    parser.add_argument("--variant", type=str, default="dev",
                        help="Dataset variant (dev or train)")
    parser.add_argument("--poison-rate", type=float, default=0.08,
                        help="Poison rate, i.e., proportion of fake evidence to total evidence")
    parser.add_argument("--attack-type", type=str, default="disinformation",
                        choices=ATTACK_TYPES,
                        help="Attack type")
    parser.add_argument("--victim",type=str, default="defame",
                        choices=VICTIM_SYSTEMS,
                        help="Fact checking system being attacked")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of claims to process for testing")
    parser.add_argument("--attacker-model", type=str, default="gpt-4o-mini",
                        help="Model used to generate attack content")
    parser.add_argument("--fact-checker-model", type=str, default="gpt-4o-mini",
                        help="Model used for fact checking")
    parser.add_argument("--n-processes", type=int, default=1,
                        help="Number of processes to parallelize")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                        help="List of GPU IDs to use, e.g., --gpu-ids 0 1 means use GPUs 0 and 1")

    # example run command
    # python attack/main.py --victim defame --attack-type fact2fiction --poison-rate 0.08 --attacker-model gpt-4o-mini --fact-checker-model gpt-4o-mini --n-processes 4 --gpu-ids 0 1 2 3
    return parser.parse_args()

# Main function modified
if __name__ == "__main__":
    # Set multi-process startup method to spawn to solve CUDA initialization issues in multi-process
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set signal handler
    signal.signal(signal.SIGTERM, lambda signo, frame: sys.exit(1))
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set experiment directory
        exp_dirs = setup_experiment_dir(args)
        
        # Set up logging
        # logger = setup_logging(exp_dirs["logs_dir"])
        logger = Logger(print_log_level="info", target_dir=exp_dirs["logs_dir"], attack=True)
        
        logger.info("="*50)
        logger.info("Knowledge Base Poisoning Attack Experiment")
        logger.info("="*50)
        logger.info(f"Experiment configuration: {json.dumps(vars(args), indent=2)}")
        logger.info(f"Experiment directory: {exp_dirs['exp_dir']}")        
        logger.info("Starting attack...")
        result = attack_all_claims(args, exp_dirs, logger)
        
        if result:
            logger.info("Attack experiment successfully completed")
        else:
            logger.error("Attack experiment failed")
        
        logger.info("="*50)
        logger.info("Experiment ended")
        logger.info("="*50)
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
