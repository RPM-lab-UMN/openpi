import collections
import dataclasses
import logging
import math
import pathlib
import sys
from typing import Dict, List, Tuple, Optional

import imageio
import numpy as np
import pandas as pd
import torch
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Add active_testing to path if needed
# Assuming active_testing is a sibling directory to openpi
ACTIVE_TESTING_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "active_testing"
if ACTIVE_TESTING_PATH.exists() and str(ACTIVE_TESTING_PATH) not in sys.path:
    sys.path.insert(0, str(ACTIVE_TESTING_PATH))

from testers import ActiveTester, IIDSampler
from utils import fit_surrogate_model

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Set up torch device and data type
tkwargs = {"dtype": torch.double, "device": "cuda" if torch.cuda.is_available() else "cpu"}


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO-Plus environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    mapping_dir: str = "data/libero_plus/mappings"  # Directory containing pre-generated task-factor mappings

    #################################################################################################################
    # Active testing parameters
    #################################################################################################################
    num_evals_per_task: int = 20  # Number of configurations to evaluate per task (should be < total configs)
    num_init_pts: int = 5  # Number of initial random points before switching to active (per task)
    model_name: str = "SingleTaskGP"  # Surrogate model name
    acq_func_name: str = "PSD"  # Acquisition function name

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"  # Path to save videos
    output_file: str = "results/active_libero_plus_results.csv"  # Path to save results CSV
    seed: int = 7  # Random Seed (for reproducibility)


def load_task_configurations(mapping_dir: str, suite_name: str, base_task: str) -> pd.DataFrame:
    """
    Load all factor configurations for a specific base task.
    
    Returns a DataFrame with all configurations (factor combinations) for the given task.
    """
    per_task_dir = pathlib.Path(mapping_dir) / suite_name
    if not per_task_dir.exists():
        raise FileNotFoundError(
            f"Mapping directory not found: {per_task_dir}\n"
            f"Please run parse_libero_plus_factors.py first to generate mappings."
        )
    
    base_task_normalized = base_task.lower().replace(' ', '_').strip()
    safe_filename = base_task_normalized.replace('/', '_').replace('\\', '_')[:100]
    task_file = per_task_dir / f"{safe_filename}.csv"
    
    if task_file.exists():
        df = pd.read_csv(task_file)
        logging.info(f"Loaded {len(df)} configurations for task: {base_task}")
        return df
    
    # If exact match not found, try to find by searching all files
    # (in case normalization doesn't match exactly)
    logging.warning(f"Exact match not found for task: {base_task}, searching all files...")
    for task_file in per_task_dir.glob("*.csv"):
        df = pd.read_csv(task_file)
        # Check if any base_task matches
        df_normalized = df['base_task'].str.lower().str.replace(' ', '_').str.strip()
        matching_mask = (df_normalized == base_task_normalized)
        if matching_mask.any():
            # Filter to only include rows where base_task matches
            df_filtered = df[matching_mask].copy()
            logging.info(f"Found matching task in {task_file.name}")
            logging.info(f"Loaded {len(df_filtered)} configurations for task: {base_task}")
            return df_filtered
    
    logging.warning(f"No configurations found for task: {base_task}")
    return pd.DataFrame()


# Note: parse_task_name is now in parse_libero_plus_factors.py
# We don't need it here since we use pre-generated mappings


def encode_categorical_factors(
    mapping_df: pd.DataFrame, factor_columns: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Encode categorical factor values to numerical values in [0, 1].
    
    **Encoding Rationale:**
    The perturbation factors (e.g., table_1, table_2, initstate_351, view_0_0_100_0_0) are
    categorical variables - they represent discrete choices with no inherent numerical ordering.
    However, our surrogate models (Gaussian Processes) require continuous numerical inputs.
    
    We use **label encoding with normalization to [0, 1]**:
    - Each unique factor value is assigned a number (0, 1, 2, ...)
    - These are then normalized to [0, 1] by dividing by (num_values - 1)
    - Example: table_1, table_2, table_3 → 0.0, 0.5, 1.0
    
    **Why this approach:**
    1. **GP compatibility**: GPs work with continuous inputs and can learn smooth functions
       over the encoded space, allowing interpolation between factor values
    2. **Simplicity**: Single dimension per factor (vs. one-hot encoding which would create
       many dimensions)
    3. **Active learning**: The surrogate model can identify regions of high uncertainty
       between encoded values, guiding exploration
    
    **Limitations:**
    - Imposes an artificial ordering (table_1 < table_2 < table_3), which may not reflect
      actual relationships between factor values
    - For factors with many values, the spacing becomes very small, which can affect GP
      kernel behavior
    - Alternative approaches (one-hot encoding, learned embeddings) could be used but
      would require more complex models
    
    Args:
        mapping_df: DataFrame with task names and factor columns
        factor_columns: List of column names that contain factor values
    
    Returns:
        encoding_maps: Dict mapping factor_type -> {factor_value: encoded_value}
        factor_order: Dict mapping factor_type -> ordered list of unique values
    """
    encoding_maps = {}
    factor_order = {}
    
    for factor_col in factor_columns:
        if factor_col not in mapping_df.columns:
            continue
        
        # Get unique factor values
        # Sort to ensure consistent encoding (alphabetical/numerical order)
        unique_values = sorted(mapping_df[factor_col].dropna().unique())
        factor_order[factor_col] = unique_values
        
        # Encode to [0, 1] range
        encoding_map = {}
        if len(unique_values) == 1:
            # Single value, encode as 0.5 (middle of range)
            encoding_map[unique_values[0]] = 0.5
        else:
            # Multiple values, encode evenly spaced in [0, 1]
            # This creates equal spacing: first value = 0.0, last value = 1.0
            for idx, value in enumerate(unique_values):
                encoding_map[value] = idx / (len(unique_values) - 1) if len(unique_values) > 1 else 0.5
        
        encoding_maps[factor_col] = encoding_map
        logging.info(f"Encoded {factor_col}: {len(unique_values)} values → [0.0, 1.0]")
        if len(unique_values) <= 5:
            # Show encoding for small sets
            logging.info(f"  Encoding: {encoding_map}")
    
    return encoding_maps, factor_order


def find_config_by_factor_vector(
    factor_vector: torch.Tensor, design_space: torch.Tensor, config_metadata: List[Dict]
) -> Optional[Dict]:
    """Find configuration metadata matching a factor vector."""
    # Find closest match (exact match preferred)
    distances = torch.norm(design_space - factor_vector.unsqueeze(0), dim=1)
    min_idx = torch.argmin(distances).item()
    
    # Check if it's an exact match (within tolerance)
    if distances[min_idx] < 1e-6:
        return config_metadata[min_idx]
    else:
        logging.warning(f"No exact match found for factor vector {factor_vector}, using closest match")
        return config_metadata[min_idx]


def run_evaluation_rollout(
    env, task_description: str, client, args: Args, max_steps: int, initial_state=None
) -> Tuple[bool, int]:
    """
    Run a single evaluation rollout.
    
    Returns:
        success: Boolean indicating if task was completed
        steps_taken: Number of steps taken
    """
    obs = env.reset()
    action_plan = collections.deque()
    
    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    
    t = 0
    success = False
    
    while t < max_steps + args.num_steps_wait:
        try:
            # Wait for objects to stabilize
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            
            # Get preprocessed image
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            
            if not action_plan:
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                
                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])
            
            action = action_plan.popleft()
            
            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            # Increment step counter after executing the step (before checking done)
            # This ensures successful steps are counted correctly
            t += 1
            if done:
                success = True
                break
            
        except Exception as e:
            logging.error(f"Caught exception during rollout: {e}")
            break
    
    # steps_taken should always reflect the actual number of steps taken (t)
    # regardless of whether we succeeded or failed
    steps_taken = t
    return success, steps_taken


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_design_space_for_task(
    task_configs_df: pd.DataFrame, encoding_maps: Dict[str, Dict[str, float]], 
    factor_columns: List[str]
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Build design space for a single task from its configurations.
    
    Args:
        task_configs_df: DataFrame with configurations for one task (from load_task_configurations)
        encoding_maps: Maps factor columns to {factor_value: encoded_value}
        factor_columns: List of factor column names
    
    Returns:
        design_space_tensor: [N, D] tensor of encoded factor values
        config_metadata: List of dicts with task_name, factor_values for each config
    """
    design_space_list = []
    config_metadata = []
    
    for _, row in task_configs_df.iterrows():
        factor_vector = []
        factor_values_dict = {}
        
        for factor_col in factor_columns:
            if factor_col in row and pd.notna(row[factor_col]):
                factor_value = str(row[factor_col])
                factor_values_dict[factor_col] = factor_value
                if factor_col in encoding_maps and factor_value in encoding_maps[factor_col]:
                    encoded_value = encoding_maps[factor_col][factor_value]
                else:
                    encoded_value = 0.5
                    logging.warning(f"Factor value {factor_value} not in encoding map for {factor_col}")
            else:
                encoded_value = 0.5
                factor_values_dict[factor_col] = None
            
            factor_vector.append(encoded_value)
        
        design_space_list.append(factor_vector)
        config_metadata.append({
            'task_name': row['task_name'],
            'base_task': row.get('base_task', ''),
            'factor_values': factor_values_dict
        })
    
    if not design_space_list:
        raise ValueError("No valid configurations found for task!")
    
    design_space_tensor = torch.tensor(design_space_list, **tkwargs)
    return design_space_tensor, config_metadata


def eval_libero_plus_active(args: Args) -> None:
    """
    Main function for per-task active evaluation on Libero-Plus.
    
    For each task in the suite, this function:
    1. Loads all possible factor configurations for that task
    2. Runs active testing to select which configurations to evaluate
    3. Evaluates the selected configurations
    """
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine factor columns by loading a sample task configuration file
    # All tasks in a suite should have the same factor columns
    per_task_dir = pathlib.Path(args.mapping_dir) / args.task_suite_name
    if not per_task_dir.exists():
        raise FileNotFoundError(
            f"Mapping directory not found: {per_task_dir}\n"
            f"Please run parse_libero_plus_factors.py first to generate mappings."
        )
    
    # Find any task file to determine factor columns
    task_files = list(per_task_dir.glob("*.csv"))
    if not task_files:
        raise ValueError(f"No task mapping files found in {per_task_dir}")
    
    sample_df = pd.read_csv(task_files[0])
    metadata_cols = {'task_name', 'base_task', 'category'}
    factor_columns = [col for col in sample_df.columns if col not in metadata_cols]
    if not factor_columns:
        raise ValueError("No factor columns found in mapping files. Run parse_libero_plus_factors.py first.")
    
    logging.info(f"Using factor columns: {factor_columns}")
    
    # Build a combined dataframe from all task files for consistent encoding
    # This ensures all factor values across all tasks are encoded consistently
    all_task_dfs = []
    for task_file in task_files:
        df = pd.read_csv(task_file)
        all_task_dfs.append(df)
    
    combined_df = pd.concat(all_task_dfs, ignore_index=True)
    
    # Encode categorical factors (using all data to get consistent encoding)
    encoding_maps, factor_order = encode_categorical_factors(combined_df, factor_columns)
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}. Available: {list(benchmark_dict.keys())}")
    
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name} with {task_suite.n_tasks} tasks")
    
    # Set up bounds for factors (all in [0, 1] after encoding)
    bounds = torch.tensor([[0.0] * len(factor_columns), [1.0] * len(factor_columns)], **tkwargs)
    
    # Create output directory
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Determine max_steps based on task suite
    if "spatial" in args.task_suite_name.lower():
        max_steps = 220
    elif "object" in args.task_suite_name.lower():
        max_steps = 280
    elif "goal" in args.task_suite_name.lower():
        max_steps = 300
    elif "10" in args.task_suite_name.lower():
        max_steps = 520
    elif "90" in args.task_suite_name.lower():
        max_steps = 400
    else:
        max_steps = 300  # Default
        logging.warning(f"Unknown task suite, using default max_steps={max_steps}")
    
    # Initialize policy client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # Results storage
    results_data = []
    
    # Check if we're resuming from existing results
    if pathlib.Path(args.output_file).exists():
        try:
            existing_df = pd.read_csv(args.output_file)
            if not existing_df.empty:
                results_data = existing_df.to_dict('records')
                logging.info(f"Resuming from {len(results_data)} existing evaluations")
        except Exception as e:
            logging.warning(f"Could not load existing results: {e}")
    
    # Main loop: iterate through each task in the suite
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting per-task active evaluation")
    logging.info(f"Evaluations per task: {args.num_evals_per_task}")
    logging.info(f"Initial random samples per task: {args.num_init_pts}")
    logging.info(f"{'='*60}\n")
    
    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        task_description = task.language
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Task {task_id + 1}/{task_suite.n_tasks}: {task_description}")
        logging.info(f"{'='*60}")
        
        # Load all configurations for this task
        task_configs_df = load_task_configurations(args.mapping_dir, args.task_suite_name, task_description)
        
        if task_configs_df.empty:
            logging.warning(f"No configurations found for task: {task_description}")
            continue
        
        total_configs = len(task_configs_df)
        if args.num_evals_per_task >= total_configs:
            # If user wants to evaluate more than available, evaluate all configurations
            # For very few configs (1-2), we evaluate all. For more, we cap at total_configs.
            num_evals = total_configs
            if total_configs > 1:
                logging.info(
                    f"num_evals_per_task ({args.num_evals_per_task}) >= total configs ({total_configs}). "
                    f"Evaluating all {total_configs} configurations."
                )
        else:
            num_evals = args.num_evals_per_task
        
        logging.info(f"Total configurations: {total_configs}, Evaluating: {num_evals}")
        
        # Build design space for this task
        design_space, config_metadata = build_design_space_for_task(
            task_configs_df, encoding_maps, factor_columns
        )
        
        # Check for existing evaluations for this task
        task_results = [r for r in results_data if r.get('task_id') == task_id]
        loop_start_index = len(task_results)
        
        if loop_start_index >= num_evals:
            logging.info(f"Task {task_id} already has {loop_start_index} evaluations. Skipping.")
            continue
        
        # Initialize sampler for this task
        # If we have enough existing results (>= num_init_pts), use ActiveTester
        # Otherwise, start with IIDSampler and switch when we reach num_init_pts
        sampler = None
        if loop_start_index >= args.num_init_pts:
            # Resuming after initial random phase - use ActiveTester
            logging.info(f"\nResuming task {task_id} with {loop_start_index} existing evaluations. Using Active Testing.")
            
            # Prepare training data from existing evaluations
            initial_X_list = []
            initial_Y_list = []
            for row in task_results:
                # Handle NaN values: when CSV is read, None values become NaN
                # row.get() only provides default if key is missing, not if value is NaN
                factor_vector = []
                for d in range(len(factor_columns)):
                    val = row.get(f'factor_{d}', 0.5)
                    # Check if value is NaN (pd.isna handles float('nan'), numpy.nan, pd.NA, None)
                    if pd.isna(val):
                        val = 0.5
                    factor_vector.append(val)
                initial_X_list.append(torch.tensor(factor_vector, **tkwargs))
                outcome = 1.0 if row.get('outcome', 0) == 1.0 else 0.0
                initial_Y_list.append(torch.tensor([outcome], **tkwargs))
            
            if initial_X_list:
                train_X = torch.stack(initial_X_list)
                train_Y = torch.stack(initial_Y_list)
                sampler = ActiveTester(
                    train_X, train_Y, bounds, design_space, mc_points=None,
                    model_name=args.model_name, acq_func_name=args.acq_func_name
                )
            else:
                raise ValueError("No initial data available for active testing")
        
        # Per-task evaluation loop
        for i in range(loop_start_index, num_evals):
            current_mode = 'initial_random' if i < args.num_init_pts else 'active'
            
            # Switch to active testing when we reach num_init_pts (if not already using it)
            if i == args.num_init_pts and (sampler is None or isinstance(sampler, IIDSampler)):
                logging.info(f"\nReached {args.num_init_pts} initial points. Switching to Active Testing for task {task_id}.")
                
                # Prepare initial training data from previous evaluations of this task
                initial_X_list = []
                initial_Y_list = []
                for row in task_results:
                    # Handle NaN values: when CSV is read, None values become NaN
                    # row.get() only provides default if key is missing, not if value is NaN
                    factor_vector = []
                    for d in range(len(factor_columns)):
                        val = row.get(f'factor_{d}', 0.5)
                        # Check if value is NaN (pd.isna handles float('nan'), numpy.nan, pd.NA, None)
                        if pd.isna(val):
                            val = 0.5
                        factor_vector.append(val)
                    initial_X_list.append(torch.tensor(factor_vector, **tkwargs))
                    outcome = 1.0 if row.get('outcome', 0) == 1.0 else 0.0
                    initial_Y_list.append(torch.tensor([outcome], **tkwargs))
                
                if initial_X_list:
                    train_X = torch.stack(initial_X_list)
                    train_Y = torch.stack(initial_Y_list)
                    sampler = ActiveTester(
                        train_X, train_Y, bounds, design_space, mc_points=None,
                        model_name=args.model_name, acq_func_name=args.acq_func_name
                    )
                else:
                    raise ValueError("No initial data available for active testing")
            
            # Initialize IID sampler if not yet initialized
            if sampler is None:
                sampler = IIDSampler(design_space)
            
            logging.info(f"\nTask {task_id}, Config {i+1}/{num_evals} (mode: {current_mode})")
            
            # Set seed for reproducibility
            if current_mode == 'active':
                torch.manual_seed(task_id * 1000 + i + 1)
            
            # Get next configuration to evaluate
            factor_vector = sampler.get_next_point()
            
            # Find corresponding configuration
            distances = torch.norm(design_space - factor_vector.unsqueeze(0), dim=1)
            min_idx = torch.argmin(distances).item()
            config = config_metadata[min_idx]
            task_name = config['task_name']
            
            logging.info(f"Evaluating: {task_name}")
            logging.info(f"Factor values: {config['factor_values']}")
            
            # Initialize environment for this task
            env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            
            # Get initial states for this task
            initial_state = None
            try:
                initial_states = task_suite.get_task_init_states(task_id)
                if initial_states and len(initial_states) > 0:
                    initial_state = initial_states[np.random.randint(0, len(initial_states))]
            except Exception as e:
                logging.warning(f"Could not get initial states for task {task_id}: {e}")
            
            # Run evaluation rollout
            try:
                success, steps_taken = run_evaluation_rollout(
                    env, task_description, client, args, max_steps, initial_state=initial_state
                )
                outcome = 1.0 if success else 0.0
                logging.info(f"Outcome: {'Success' if success else 'Failure'} ({steps_taken} steps)")
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")
                success = False
                outcome = 0.0
                steps_taken = max_steps
            
            # Update sampler with new data
            outcome_tensor = torch.tensor([outcome], **tkwargs)
            sampler.update(factor_vector, outcome_tensor)
            
            # Record results
            entry = {
                'task_id': task_id,
                'task_description': task_description,
                'config_idx': i + 1,
                'mode': current_mode,
                'task_name': task_name,
                'outcome': outcome,
                'success': success,
                'steps_taken': steps_taken,
            }
            # Add factor values
            for dim_idx in range(len(factor_columns)):
                entry[f'factor_{dim_idx}'] = factor_vector[dim_idx].item()
                entry[f'factor_{factor_columns[dim_idx]}'] = config['factor_values'].get(factor_columns[dim_idx], None)
            
            results_data.append(entry)
            task_results.append(entry)
            
            # Save results incrementally
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(args.output_file, index=False)
            
            env.close()
        
        # Log task statistics
        task_successes = sum(1 for r in task_results if r.get('outcome', 0) == 1.0)
        task_total = len(task_results)
        if task_total > 0:
            logging.info(f"\nTask {task_id} complete: {task_successes}/{task_total} successes ({task_successes/task_total*100:.1f}%)")
        else:
            logging.info(f"\nTask {task_id} complete: No evaluations performed (skipped)")
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("Evaluation complete!")
    if results_data:
        successes = sum(1 for r in results_data if r.get('outcome', 0) == 1.0)
        total = len(results_data)
        logging.info(f"Total success rate: {successes}/{total} ({successes/total*100:.1f}%)")
        logging.info(f"Total evaluations: {total}")
    logging.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_plus_active)
