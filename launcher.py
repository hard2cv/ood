"""
DDP:
python launcher.py --dataset coco --eval_model LLaVA-NeXT-8B --nproc_per_node 2 --master_port 10000 --cuda_devices "0,1" --batchsize 1 --ddp --is_ood
"""
from config import create_arg_parser
import importlib
import os, sys
import json
import models
import logging
import subprocess


def run_model_in_env(env_path, module_args, args):
    """Run the model evaluation in the specified virtual environment"""
    # Constructing the interpreter path
    python_exec = os.path.join(env_path, "bin", "python") if not args.use_deepspeed else os.path.join(env_path, "bin", "deepspeed")
    print('python_exec:', python_exec)
    
    # Construct full command arguments (list format recommended to avoid shell injection risks)
    if args.use_deepspeed:
        cmd = [
            python_exec,
            f"--include=localhost:{args.cuda_devices}",
            "path/to/models/run_models.py"
        ] + module_args
    else:
        cmd = [
            python_exec,
            "-u",
            *(
            ["-m", "torch.distributed.launch", "--use_env",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}"]
            if args.ddp else []
        ),
            "path/to/models/run_models.py"
        ] + module_args
    
 
    # Set CUDA_VISIBLE_DEVICES in subprocess environment
    if not args.use_deepspeed:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge error output to standard output
            text=True,
            bufsize=1,  # row buffer mode
            # env=env,
            encoding="utf-8",
            **({} if args.use_deepspeed else {"env": env})
        )

        # Real-time reading of output streams
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logging.info(output.strip())

        # Checking the final exit status
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, 
                cmd, 
                output="",
                stderr=""
            )

    except subprocess.CalledProcessError as e:
        error_msg = f"subprocess failure [code={e.returncode}]"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def main():
    # Load model configurations from JSON
    with open(args.model_config, "r") as f:
        models_config = json.load(f)

    # Retrieve the module path and environment path for the selected model
    model_config = models_config[args.eval_model]

    module_args = [
        "--dataset", args.dataset,
        "--eval_model", args.eval_model,
        "--batchsize", str(args.batchsize),
        "--is_ood" if args.is_ood else None,
        "--ddp" if args.ddp else None,
        "--use_deepspeed" if args.use_deepspeed else None
        ]
    
    # Remove None values (this step ensures that only the necessary parameters are included)
    module_args = [arg for arg in module_args if arg is not None]

    run_model_in_env(
        env_path=model_config["env_path"],
        module_args=module_args,
        args=args
    )

if __name__ == "__main__":
    # parameters init
    parser = create_arg_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    main()
