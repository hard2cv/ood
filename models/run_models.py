import numpy as np
import torch
import torch.distributed as dist
import sys, os
import argparse
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import create_arg_parser, DATASETS_CSV_PATH

from scripts.utils import (
    check_dataset_and_model as check_dataset_and_model,
    InvalidDatasetOrModelError as InvalidDatasetOrModelError,
    get_evaluation_log_path as get_evaluation_log_path,
    setup_logging as setup_logging,
    log_args as log_args
)

def select_and_run_model(args):
    match args.eval_model:
        case "DeepSeek-VL-7B-Chat" | "DeepSeek-VL2-Small":
            from DeepSeekVLSeries import run_model
        case 'InternVL2-8B' | 'InternVL2_5-8B':
            from InternVLSeries import run_model
        case 'Qwen2-VL-7B-Instruct' | 'Qwen2_5-VL-7B-Instruct':
            from QwenSeries import run_model
        case 'Llama-3.2-11B-Vision-Instruct':
            from LlamaVision import run_model
        case 'LLaVA-NeXT-8B':
            from LLaVA_NeXT import run_model
        case 'GPT-4o':
            from GPTSeries import run_model
        case 'Gemini':
            from GeminiSeries import run_model
        case _:
            raise ValueError(f"Unsupported model: {args.eval_model}")
    run_model(args)

def main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Main function to run the evaluation process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        parser (argparse.ArgumentParser): Argument parser instance.
    """
    try:
        # Validate dataset and model parameters
        args.dataset_path, args.model_para_path, args.module = check_dataset_and_model(args, parser)
    except InvalidDatasetOrModelError as e:
        logging.error(f"Execution terminated due to invalid dataset or model: {e}")
        return  # Stop execution if dataset or model is invalid

    # Initialize logging system
    get_evaluation_log_path(args)
    setup_logging(args)
    
    logging.info("Starting evaluation process...")

    # Log all arguments for reproducibility
    log_args(args)
    logging.info(f"Logging directory: {args.log_dir}")

    try:
        # Select and run the model
        select_and_run_model(args)
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {e}", exc_info=True)
        return  # Terminate execution on failure

    logging.info("Evaluation process completed successfully.")


if __name__ == "__main__":
    # parameters init
    parser = create_arg_parser()
    args = parser.parse_args()

    logging.info(f'DDP: {args.ddp}')
    print(f'DDP: {args.ddp}')

    if args.ddp:
        dist.init_process_group(backend='nccl')  # Using the NCCL Backend
        local_rank = dist.get_rank()  # Get the ranking of the current process
        torch.cuda.set_device(local_rank)  # Setting the GPU device for the current process
        args.local_rank = local_rank
        args.nproc_per_node = torch.cuda.device_count()
    else:
        args.local_rank = 0

    args.dataset_csv_path = DATASETS_CSV_PATH[args.dataset]['ood' if args.is_ood else 'id']

    # Set not to use scientific notation and set print accuracy
    torch.set_printoptions(precision=10, sci_mode=False)
    # Setting NumPy printing options to disable scientific notation
    np.set_printoptions(precision=10, suppress=True)

    main(args, parser)

    if dist.is_initialized():
        dist.destroy_process_group()
