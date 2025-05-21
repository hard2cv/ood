import logging
import os, sys
from datetime import datetime
import json
import re
import torch
import base64
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score as accuracy_score, 
    f1_score as f1_score, 
    precision_score as precision_score, 
    recall_score as recall_score, 
    roc_auc_score as roc_auc_score, 
    matthews_corrcoef as matthews_corrcoef
    )

class InvalidDatasetOrModelError(Exception):
    def __init__(self, message):
        super().__init__(message)


def check_dataset_and_model(args, parser):
    """
    Checks if the given dataset and model are in the allowed lists. 
    If they are, returns the dataset path and model parameter path; 
    otherwise, raises an exception.
    
    Args:
        args: Additional arguments that might be needed for further validation.
        dataset_name (str): The name of the dataset to be validated.
        model_name (str): The name of the model to be validated.
    
    Raises:
        InvalidDatasetOrModelError: If the dataset or model is not found in the respective allowed list.
    
    Returns:
        triple: A triple containing the dataset path, model parameter path and module.
    """
    # Load model configurations from JSON
    with open(args.model_config, "r") as f:
        models_config = json.load(f)

    choices_dict = {action.dest: action.choices for action in parser._actions}

    # Check if the dataset is in the DATASETS dictionary
    if args.dataset not in args.DATASET_PATH:
        logging.critical(f"Error: Dataset '{args.dataset}' is not in the allowed DATASETS list.")
        raise InvalidDatasetOrModelError(f"Dataset '{args.dataset}' is not in the allowed DATASETS list.")
    
    # Check if the model is in the MODELS dictionary
    if args.eval_model not in choices_dict.get('eval_model'):
        logging.critical(f"Error: Model '{args.eval_model}' is not in the allowed MODELS list.")
        raise InvalidDatasetOrModelError(f"Model '{args.eval_model}' is not in the allowed MODELS list.")

    logging.info("Dataset and Model are valid.")
    return args.DATASET_PATH.get(args.dataset), models_config[args.eval_model]['para_path'], models_config[args.eval_model]['module']

def logging_file_name(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M')
    args.timestamp = formatted_time
    
    log_file = \
        f'{formatted_time}_{args.dataset}_{args.eval_model}_{"ood" if args.is_ood else "id"}_data{"CoT.log" if args.enable_cot else ".log"}'

    return log_file

def get_evaluation_log_path(args):
    """
    Generate the logging path based on dataset name, model name, test data type, CoT flag, and evaluation log directory.

    The function constructs a structured logging path using the following components:
    - `log_dir`: Base directory for storing logs.
    - `cot_flag`: If `enable_cot` is True, add an intermediate "cot" directory.
    - `test_data_type`: Specifies whether the test is on OOD or ID data.
    - `dataset`: Name of the dataset being evaluated.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - log_dir (str): Base directory for logs.
            - enable_cot (bool): Flag indicating whether Chain-of-Thought (CoT) is enabled.
            - is_ood (bool): Flag indicating whether testing is on OOD data.
            - dataset (str): Name of the dataset.

    Returns:
        str: The full path to the logging directory.

    Raises:
        ValueError: If the constructed evaluation_logs_path is invalid.
    """
    log_file_name = logging_file_name(args)

    # Determine if the "cot" directory should be included
    cot_segment = ["cot"] if args.enable_cot else []

    # Construct the evaluation logs path
    evaluation_logs_path = os.path.join(args.log_dir, *cot_segment, 'ood' if args.is_ood else 'id', args.dataset, log_file_name)

    # Validate the logging path
    if not evaluation_logs_path:
        raise ValueError("The constructed evaluation_logs_path is invalid.")
    
    args.log_dir = evaluation_logs_path

def setup_logging(args):
    log_file = args.log_dir
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Getting the Root Logger and Resetting the Configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove all existing Handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file Handler to ensure that the log file is written to the child process
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Create a console handler that outputs to stdout for the parent process to catch.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Add Handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Test Log
    logger.info("=== Subprocess logging configured successfully ===")

def log_args(args):
    """
    Log all important arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logging.info("===== Experiment Configuration =====")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info("===================================")


def check_data_csv(dataset_name, test_data_type, hard_or_simple, DATASETS_CSV_PATH):
    ID_data, OOD_data_Y_N, OOD_YN_pair = None, None, None
    
    simple_hard_or_pair_dict = DATASETS_CSV_PATH.get(dataset_name)

    if test_data_type == 'id':
        ID_data = simple_hard_or_pair_dict.get(test_data_type)
    elif test_data_type == 'ood':
        if hard_or_simple == 'simple' or hard_or_simple == 'hard':
            OOD_data_Y_N = simple_hard_or_pair_dict.get(hard_or_simple)
        elif hard_or_simple == None:
            raise InvalidDatasetOrModelError(f"Logic error, when test_data_type variable is {test_data_type}, hard_or_simple cannot be {hard_or_simple}.")
    elif test_data_type == 'ood_yes_no_pair':
        OOD_YN_pair = simple_hard_or_pair_dict.get('yes_no_pair')

    if OOD_data_Y_N == None and OOD_YN_pair == None and ID_data == None:
        raise InvalidDatasetOrModelError(f"Logic error! ID_data is {ID_data} and OOD_data_Y_N is {OOD_data_Y_N} and OOD_YN_pair is {OOD_YN_pair}! All three cannot be None at the same time.")
    else:
        return ID_data, OOD_data_Y_N, OOD_YN_pair

def convert_responses_to_binary(response_list: List[str]) -> List[int]:
    """
    Converts natural language responses to binary values based on keyword presence.

    Processing Rules:
    1. Return 1 if the response contains 'yes' (case-insensitive whole word match) 
       without conflicting 'no'
    2. Return 0 if:
       - Response contains 'no' (case-insensitive whole word match) without 'yes'
       - Response contains both 'yes' and 'no' (ambiguous case)
       - Response contains neither keyword

    Args:
        response_list: List of string responses to analyze

    Returns:
        List of binary values (0/1) corresponding to each input response

    Examples:
        >>> convert_responses_to_binary(['Yes, confirmed', 'No matching results'])
        [1, 0]
        
        >>> convert_responses_to_binary(['Affirmative (yes)', 'Negative', 'Both yes and no'])
        [1, 0, 0]
    """
    YES_KEYWORD = re.compile(r'\byes\b', flags=re.IGNORECASE)
    NO_KEYWORD = re.compile(r'\bno\b', flags=re.IGNORECASE)
    
    binary_output = []
    
    for text in response_list:
        contains_yes = bool(YES_KEYWORD.search(text))
        contains_no = bool(NO_KEYWORD.search(text))

        if contains_yes and not contains_no:
            binary_output.append(1)
        else:  # Handles no/neither/conflict cases
            binary_output.append(0)
    
    return binary_output

def extend_results(
    gathered_data: List[Optional[Tuple[List[int], List[int], List[float], List[int], List[float], List[int]]]],
    all_preds:      torch.Tensor,
    all_labels:     torch.Tensor,
    batch_preds:    torch.Tensor,
    batch_labels:   torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extend prediction and label tensors with gathered batch data.

    Args:
        gathered_data (List[Optional[Tuple[List[int], List[int], List[float], List[int], List[float], List[int]]]]]):
            A list containing tuples with image IDs, category labels, presence response scores, 
            presence ground truth labels, absence response scores, and absence ground truth labels.
            Some entries may be None and should be ignored.
        all_preds (torch.Tensor):
            A tensor containing all accumulated model predictions.
        all_labels (torch.Tensor):
            A tensor containing all accumulated ground truth labels.
        batch_preds (torch.Tensor):
            A tensor containing batch-specific model predictions.
        batch_labels (torch.Tensor):
            A tensor containing batch-specific ground truth labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Updated all_preds, all_labels, batch_preds, and batch_labels tensors.
    """
    for data in gathered_data:
        if data is None:
            continue
        _, _, response_presence_binary, presence_label, response_absence_binary, absence_label = data

        preds = torch.tensor(response_presence_binary + response_absence_binary, dtype=torch.uint8)
        labels = torch.tensor(presence_label + absence_label, dtype=torch.uint8)

        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        batch_preds = torch.cat((batch_preds, preds), dim=0)
        batch_labels = torch.cat((batch_labels, labels), dim=0)

    return all_preds, all_labels, batch_preds, batch_labels

def extend_merge_results(
    gathered_data: list,
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    batch_preds: torch.Tensor,
    batch_labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merges prediction and label tensors from gathered data into cumulative and batch-wise results.

    Args:
        gathered_data (list): A list containing tuples of gathered data from different processes.
        all_preds (torch.Tensor): Tensor containing all accumulated predictions.
        all_labels (torch.Tensor): Tensor containing all accumulated labels.
        batch_preds (torch.Tensor): Tensor containing batch-wise predictions.
        batch_labels (torch.Tensor): Tensor containing batch-wise labels.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Updated tensors (all_preds, all_labels, batch_preds, batch_labels).
    """
    for data in gathered_data:
        if data is None:
            continue
        
        _, _, response_binary, labels = data
        
        preds = torch.as_tensor(response_binary, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.uint8)

        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        batch_preds = torch.cat((batch_preds, preds), dim=0)
        batch_labels = torch.cat((batch_labels, labels), dim=0)
    
    return all_preds, all_labels, batch_preds, batch_labels

def metric_performances(
    preds: (np.ndarray, list),
    labels: (np.ndarray, list)
) -> dict[str, float]:
    """Calculate multiple classification metrics and return as a dictionary.
    
    Args:
        preds: Array-like of predicted labels (binary). Can be numpy array or list.
        labels: Array-like of ground truth labels (binary). Can be numpy array or list.
    
    Returns:
        containing:
        - accuracy: Accuracy score
        - f1: F1 score
        - precision: Precision score
        - recall: Recall score
        - mcc: Matthews correlation coefficient
    
    Raises:
        ValueError: If inputs have different lengths
    """
    # Convert inputs to numpy arrays if they're not already
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Validate input shapes
    if len(preds) != len(labels):
        logging.error(f"Input length mismatch: preds ({len(preds)}), labels ({len(labels)})")
        raise ValueError(f"Input length mismatch: preds ({len(preds)}), labels ({len(labels)})")

    # Calculate metrics
    accuracy        = accuracy_score(labels, preds)
    f1              = f1_score(labels, preds)
    precision       = precision_score(labels, preds)
    recall          = recall_score(labels, preds)
    mcc             = matthews_corrcoef(labels, preds)

    return accuracy, f1, precision, recall, mcc

def format_elapsed_time(start_time: float, end_time: float) -> str:
    """
    Format the elapsed time between start_time and end_time into HH:MM:SS format.
    
    Parameters:
        start_time (float): The start timestamp, typically obtained from time.time().
        end_time (float): The end timestamp, typically obtained from time.time().
    
    Returns:
        str: The formatted elapsed time as a string in "HH:MM:SS" format.
    
    Raises:
        ValueError: If end_time is earlier than start_time.
    """
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time.")
    
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def store_tensor(tensor: torch.Tensor, path: str) -> None:
    """
    Save a torch.Tensor to the specified file path.

    Parameters:
        tensor (torch.Tensor): The tensor to be saved.
        path (str): The file path where the tensor will be saved.

    Raises:
        ValueError: If the input is not a torch.Tensor.
        RuntimeError: If an error occurs during saving.
    """
    # Validate that the input is a torch.Tensor
    if not isinstance(tensor, torch.Tensor):
        logging.error("The input must be a torch.Tensor.")
        raise ValueError("The input must be a torch.Tensor.")

    # Ensure the directory exists, if not, create it
    save_dir = os.path.dirname(path)
    if save_dir:  # Avoid issues if save_path is just a filename (current directory)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory '{save_dir}': {e}")
            return

    try:
        # Save the tensor to the specified path
        torch.save(tensor, path)
        logging.info(f"Tensor successfully saved to {path}.")
    except Exception as e:
        logging.error(f"An error occurred while saving the tensor: {e}")
        raise RuntimeError(f"An error occurred while saving the tensor: {e}")


def construct_tensor_path(label: bool, args) -> str:
    """
    Construct the file path for saving a tensor.

    Parameters:
        label (bool): A boolean indicating whether the tensor is for predictions (True) or labels (False).
        args (object): An object containing the required attributes:
            - timestamp (str): Timestamp of the experiment.
            - dataset (str): Name of the dataset.
            - eval_model (str): Name of the evaluation model.
            - is_ood (bool): Whether the data is out-of-distribution (OOD).
            - store_tensor_path (str, optional): Base directory for storing tensors.

    Returns:
        str: The complete file path for saving the tensor.
    """
    file_name = f"{args.timestamp}_{args.dataset}_{args.eval_model}_{'ood_data' if args.is_ood else 'id_data'}_{'preds' if label else 'labels'}.pt"
    
    # Ensure path compatibility across different operating systems
    if args.store_tensor_path:
        return os.path.join(args.store_tensor_path, file_name)
    
    return file_name

def load_module_keys(json_path: str) -> List[str]:
    """
    Reads a JSON file containing a dictionary of modules, and returns a list of all keys.

    :param json_path: Path to the JSON file.
    :return: A list containing all keys from the JSON dictionary.
    """
    try:
        # Open and read the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure the data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("The content of the JSON file must be a dictionary.")

        # Return the list of keys
        return list(data.keys())
    
    except Exception as e:
        logging.error(f"Error while processing the JSON file: {e}")
        return []

def encode_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")