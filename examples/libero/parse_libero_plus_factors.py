"""
Parse task_classification.json to create task-factor mappings for each suite.

This script processes the Libero-Plus task classification JSON file and creates
CSV files mapping each base task to its possible factor configurations.
"""

import json
import logging
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import tyro


def parse_task_name(task_name: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse Libero-Plus task name to extract base task and perturbation factors.
    
    The 7 factor types in Libero-Plus are:
    1. Background Textures: table_<num> or tb_<num>
    2. Robot Initial States: initstate_<num>
    3. Camera Viewpoints: view_<numbers>
    4. Language Instructions: language_<num>
    5. Sensor Noise: noise_<num>
    6. Objects Layout: add_<num> or level<num>_sample<num>
    7. Light Conditions: light_<num>
    
    Returns:
        base_task: Base task description (with factors removed)
        factors: Dictionary of factor_type -> factor_value
    """
    factors = {}
    base_task = task_name
    
    # Define patterns for each factor type
    # Order matters - match more specific patterns first
    # View pattern: match view_ followed by digits and underscores until next factor or end
    # Use a lookahead to stop before other factor markers
    factor_patterns = [
        # Camera Viewpoints: view_<numbers> - match full sequence (greedy) until next factor or end
        # Pattern: _view_ followed by digits and underscores, stopping before _initstate_, _language_, etc. or end
        (r'_view_(\d+(?:_\d+)*)(?=_initstate_|_language_|_noise_|_light_|_add_|_level\d+_sample|_table_|_tb_|$)', 'view'),
        # Objects Layout: level<num>_sample<num> or add_<num>
        (r'_level(\d+)_sample(\d+)', 'layout'),  # Special case: level5_sample4
        (r'_add_(\d+)', 'add'),
        # Background Textures: table_<num> or tb_<num>
        (r'_table_(\d+)', 'table'),
        (r'_tb_(\d+)', 'table'),  # Alternative format
        # Robot Initial States: initstate_<num>
        (r'_initstate_(\d+)', 'initstate'),
        # Language Instructions: language_<num>
        (r'_language_(\d+)', 'language'),
        # Sensor Noise: noise_<num>
        (r'_noise_(\d+)', 'noise'),
        # Light Conditions: light_<num>
        (r'_light_(\d+)', 'light'),
    ]
    
    # Extract all factors from the task name
    for pattern, factor_type in factor_patterns:
        matches = list(re.finditer(pattern, task_name))
        for match in matches:
            if factor_type == 'view':
                # For view, capture the full pattern including all numbers
                numbers_part = match.group(1)
                factor_value = f"view_{numbers_part}"
                factors[factor_type] = factor_value
            elif factor_type == 'layout':
                # Special handling for level<num>_sample<num>
                level = match.group(1)
                sample = match.group(2)
                factor_value = f"level{level}_sample{sample}"
                factors['layout'] = factor_value
            else:
                # For other factors, reconstruct the value
                num = match.group(1)
                if factor_type == 'table' and 'tb_' in match.group(0):
                    factor_value = f"tb_{num}"
                else:
                    factor_value = f"{factor_type}_{num}"
                factors[factor_type] = factor_value
    
    # Remove all factors from base_task to get the clean task description
    # Find the earliest position where any factor starts
    earliest_factor_start = len(base_task)
    for pattern, factor_type in factor_patterns:
        matches = list(re.finditer(pattern, base_task))
        for match in matches:
            if match.start() < earliest_factor_start:
                earliest_factor_start = match.start()
    
    # If we found factors, take everything before the first factor
    if earliest_factor_start < len(base_task):
        base_task = base_task[:earliest_factor_start]
    
    # Clean up any trailing underscores
    base_task = base_task.rstrip('_')
    
    return base_task, factors


def normalize_task_description(task_desc: str) -> str:
    """Normalize task description for matching (lowercase, replace spaces with underscores)."""
    return task_desc.lower().replace(' ', '_').strip()


def load_tasks_from_jsonl(jsonl_path: str) -> Dict[int, str]:
    """Load task descriptions from tasks.jsonl file."""
    tasks = {}
    if pathlib.Path(jsonl_path).exists():
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                tasks[data['task_index']] = data['task']
        logging.info(f"Loaded {len(tasks)} tasks from {jsonl_path}")
    return tasks


def parse_suite(
    suite_name: str,
    suite_data: List[Dict],
    tasks_jsonl_path: str = None,
    output_dir: str = "data/libero_plus/mappings"
) -> None:
    """
    Parse a task suite and create task-factor mappings.
    
    Args:
        suite_name: Name of the suite (e.g., "libero_spatial")
        suite_data: List of task entries from task_classification.json
        tasks_jsonl_path: Optional path to tasks.jsonl for matching task indices
        output_dir: Directory to save the mapping CSV files
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing suite: {suite_name}")
    logging.info(f"{'='*60}")
    
    # Load task descriptions if available
    tasks_by_index = {}
    if tasks_jsonl_path:
        tasks_by_index = load_tasks_from_jsonl(tasks_jsonl_path)
    
    # Group configurations by base task
    # Key: normalized base task description, Value: list of (full_task_name, factors, category)
    task_configs = defaultdict(list)
    
    for entry in suite_data:
        task_name = entry['name']
        category = entry.get('category', 'Unknown')
        
        # Parse task name
        base_task, factors = parse_task_name(task_name)
        
        # Normalize base task for grouping
        base_task_normalized = normalize_task_description(base_task)
        
        # Store configuration
        task_configs[base_task_normalized].append({
            'full_task_name': task_name,
            'base_task': base_task,
            'factors': factors,
            'category': category
        })
    
    logging.info(f"Found {len(task_configs)} unique base tasks")
    
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    per_task_dir = output_path / suite_name
    per_task_dir.mkdir(parents=True, exist_ok=True)
    
    # For each base task, create a CSV with all its factor configurations
    # Only include factors that vary within each category
    for base_task_norm, configs in task_configs.items():
        # Group configurations by category to determine which factors vary
        configs_by_category = defaultdict(list)
        for config in configs:
            configs_by_category[config['category']].append(config)
        
        # For each category, determine which factors vary
        # Only include factors that have multiple unique values within that category
        varying_factors_by_category = {}
        for category, category_configs in configs_by_category.items():
            # Collect all factor values for each factor type in this category
            factor_values_by_type = defaultdict(set)
            for config in category_configs:
                for factor_type, factor_value in config['factors'].items():
                    if factor_value is not None:
                        factor_values_by_type[factor_type].add(factor_value)
            
            # Only include factors that vary (have more than one unique value)
            varying_factors = {
                factor_type for factor_type, values in factor_values_by_type.items()
                if len(values) > 1
            }
            varying_factors_by_category[category] = varying_factors
        
        # Build rows for this task, only including varying factors
        task_configs_list = []
        all_varying_factor_types = set()
        
        # Collect all varying factor types across all categories for this task
        for varying_factors in varying_factors_by_category.values():
            all_varying_factor_types.update(varying_factors)
        
        for config in configs:
            row = {
                'task_name': config['full_task_name'],
                'base_task': config['base_task'],
                'category': config['category']
            }
            
            # Only include factors that vary within this category
            category = config['category']
            varying_factors = varying_factors_by_category.get(category, set())
            
            for factor_type in sorted(all_varying_factor_types):
                # Only include if this factor varies in this category
                if factor_type in varying_factors:
                    row[factor_type] = config['factors'].get(factor_type, None)
                else:
                    # Factor doesn't vary in this category, set to None
                    row[factor_type] = None
            
            task_configs_list.append(row)
        
        task_df = pd.DataFrame(task_configs_list)
        # Create a safe filename from base task
        safe_filename = base_task_norm.replace('/', '_').replace('\\', '_')[:100]  # Limit length
        task_output_file = per_task_dir / f"{safe_filename}.csv"
        task_df.to_csv(task_output_file, index=False)
        
        logging.info(f"  {configs[0]['base_task']}: {len(configs)} configurations, varying factors: {sorted(all_varying_factor_types)}")
    
    logging.info(f"\nSaved {len(task_configs)} per-task mappings to: {per_task_dir}")


def main(
    task_classification_json: str = "/data/liao0241/LIBERO-plus/libero/libero/benchmark/task_classification.json",
    tasks_jsonl_base_dir: str = None,  # Base directory for tasks.jsonl files (e.g., ~/.cache/huggingface/lerobot/iamandrewliao/)
    output_dir: str = "./examples/libero/libero_plus_mappings",
    suite_names: List[str] = None,  # If None, process all suites
) -> None:
    """
    Parse task_classification.json and create task-factor mappings.
    
    Args:
        task_classification_json: Path to task_classification.json file
        tasks_jsonl_base_dir: Base directory where tasks.jsonl files are located
            (e.g., ~/.cache/huggingface/lerobot/iamandrewliao/)
            The script will look for {suite_name}/meta/tasks.jsonl
        output_dir: Directory to save mapping CSV files
        suite_names: List of suite names to process (e.g., ["libero_spatial", "libero_object"])
            If None, processes all suites in the JSON file
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load task classification JSON
    if not pathlib.Path(task_classification_json).exists():
        raise FileNotFoundError(f"Task classification file not found: {task_classification_json}")
    
    logging.info(f"Loading task classification from: {task_classification_json}")
    with open(task_classification_json, 'r') as f:
        classification_data = json.load(f)
    
    # Get available suites
    available_suites = list(classification_data.keys())
    logging.info(f"Available suites: {available_suites}")
    
    # Determine which suites to process
    if suite_names is None:
        suite_names = available_suites
    else:
        # Validate suite names
        invalid_suites = [s for s in suite_names if s not in available_suites]
        if invalid_suites:
            raise ValueError(f"Invalid suite names: {invalid_suites}. Available: {available_suites}")
    
    # Process each suite
    for suite_name in suite_names:
        suite_data = classification_data[suite_name]
        
        # Try to find tasks.jsonl for this suite
        tasks_jsonl_path = None
        if tasks_jsonl_base_dir:
            # Try different possible paths
            possible_paths = [
                pathlib.Path(tasks_jsonl_base_dir) / suite_name / "tasks.jsonl",
                pathlib.Path(tasks_jsonl_base_dir) / suite_name / "meta" / "tasks.jsonl",
                pathlib.Path(tasks_jsonl_base_dir) / f"libero_plus_{suite_name.split('_')[-1]}" / "meta" / "tasks.jsonl",
            ]
            for path in possible_paths:
                if path.exists():
                    tasks_jsonl_path = str(path)
                    break
        
        parse_suite(suite_name, suite_data, tasks_jsonl_path, output_dir)
    
    logging.info("\n" + "="*60)
    logging.info("Parsing complete!")
    logging.info("="*60)


if __name__ == "__main__":
    tyro.cli(main)
