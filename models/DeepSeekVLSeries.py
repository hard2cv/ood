import os, sys
import torch
import time
from tqdm import tqdm
import csv
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import random
from torch.utils.data import Dataset
import torch.distributed as dist
from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary, 
    extend_results as extend_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor
    )
import torch.nn as nn
from typing import Any, Tuple
import deepspeed

from transformers import AutoModelForCausalLM


base_conversation = [
        {"role": "User", "content": "Compare and contrast <image_placeholder>", "images": ["path/to/image"]},
        {"role": "Assistant", "content": ""}
    ]


def deepseek_vl_collate_fn(batch):
    """
    Collate function for DeepSeek-VL dataset batches.
    
    Processes and batches different types of data from individual samples:
    - Stacks visual inputs into tensors
    - Preserves list structure for metadata
    - Handles both presence and absence modality data
    
    Args:
        batch: List of dataset items containing dictionary entries
    
    Returns:
        Dictionary containing batched tensors and metadata lists
    """
    # Batch visual inputs by stacking tensors
    inputs_presence = [item['inputs_presence'] for item in batch][0]
    inputs_absence = [item['inputs_absence'] for item in batch][0]
    
    # Collect metadata while preserving list structure
    metadata = {
        'img_id'                        : [item['img_id'] for item in batch],
        'category'                      : [item['category'] for item in batch],
        'presence_label'                : [item['presence_label'] for item in batch],
        'absence_label'                 : [item['absence_label'] for item in batch],
        'presence_question_template'    : [item['presence_question_template'] for item in batch],
        'absence_question_template'     : [item['absence_question_template'] for item in batch]
    }
    
    return {
        'inputs_presence': inputs_presence,
        'inputs_absence': inputs_absence,
        **metadata
    }

class DeepSeekVL7BChatDataset(Dataset):
    """
    Dataset for DeepSeek-VL-7B-Chat model evaluation with OOD detection.
    
    Args:
        csv_file_path (str): Path to the CSV file containing image-category-label pairs.
        datasets_img_path (str): Path to the dataset images.
        vl_chat_processor (object): Processor for handling vision-language inputs.
        vl_gpt (object): Vision-language model instance.
    """
    def __init__(self, args, vl_chat_processor, vl_gpt):
        if not os.path.exists(args.dataset_csv_path):
            raise FileNotFoundError(f"CSV file not found: {args.dataset_csv_path}")
        if not os.path.exists(args.DATASET_PATH.get(args.dataset)):
            raise FileNotFoundError(f"Dataset image path not found: {args.DATASET_PATH.get(args.dataset)}")
        self.args = args
        self.img_category_pairs = []  # Stores image-category-label tuples
        self.datasets_img_path = args.DATASET_PATH.get(args.dataset)
        self.processor = vl_chat_processor
        self.vl_gpt = vl_gpt

        if "VL2" in self.args.eval_model:
            from deepseek_vl2.utils.io import load_pil_images
            self.load_pil_images = load_pil_images
        else:
            from deepseek_vl.utils.io import load_pil_images
            self.load_pil_images = load_pil_images
        
        # Load image-category-label pairs
        self._load_img_label_pairs(args.dataset_csv_path)

        self.base_conversation = [
            {"role": "User", "content": "Compare and contrast <image_placeholder>", "images": ["path/to/image"]},
            {"role": "Assistant", "content": ""}
        ] if "VL2" not in args.eval_model else [
            {"role": "<|User|>", "content": "Text <image>", "images": ["path/to/image"]},
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    def _load_img_label_pairs(self, csv_file_path):
        """Loads image-category-label pairs from the CSV file."""
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                img_id, category, label = row[0], row[1], row[2]
                self.img_category_pairs.append((img_id, category, 1 if label.upper() == 'YES' else 0))
    
    def __len__(self):
        return len(self.img_category_pairs)
    
    def __getitem__(self, idx):
        img_id, category, presence_label = self.img_category_pairs[idx]
        absence_label = 1 - presence_label
        
        # Determine image path
        img_path = os.path.join(self.datasets_img_path, img_id)

        if self.args.dataset == 'cityscapes':
            if 'frankfurt' in img_id:
                img_path = os.path.join(self.datasets_img_path, 'frankfurt', img_id)
            elif 'lindau' in img_id:
                img_path = os.path.join(self.datasets_img_path, 'lindau', img_id)
            elif 'munster' in img_id:
                img_path = os.path.join(self.datasets_img_path, 'munster', img_id)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Generate prompts
        presence_question_template = self.args.presence_question_template.replace('[class]', category)
        absence_question_template = self.args.absence_question_template.replace('[class]', category)

        # Create conversation copies
        messages_presence = copy.deepcopy(self.base_conversation)
        messages_absence = copy.deepcopy(self.base_conversation)
        
        messages_presence[0].update(
            {"content": f"<image_placeholder>{presence_question_template}", "images": [img_path]} 
            if "VL2" not in self.args.eval_model else 
            {"content": f"{presence_question_template}<image>", "images": [img_path]}
            )
        messages_absence[0].update(
            {"content": f"<image_placeholder>{absence_question_template}", "images": [img_path]} 
            if "VL2" not in self.args.eval_model else 
            {"content": f"{absence_question_template}<image>", "images": [img_path]}
            )
        
        # Process images
        pil_images_presence = self.load_pil_images(messages_presence)
        pil_images_absence = self.load_pil_images(messages_absence)
        
        inputs_presence = self.processor(conversations=messages_presence, images=pil_images_presence, force_batchify=True).to(self.vl_gpt.device)
        inputs_absence = self.processor(conversations=messages_absence, images=pil_images_absence, force_batchify=True).to(self.vl_gpt.device)
        
        return {
            'inputs_presence'               : inputs_presence,
            'inputs_absence'                : inputs_absence,
            'img_id'                        : img_id,
            'category'                      : category,
            'presence_label'                : presence_label,
            'absence_label'                 : absence_label,
            'presence_question_template'    : presence_question_template,
            'absence_question_template'     : absence_question_template
        }

def deepseekvl_forward(
    args,
    vl_gpt, 
    tokenizer: Any, 
    inputs
) -> torch.Tensor:
    """
    Forward function for DeepSeek-VL model to generate responses from image-text inputs.
    
    Args:
        vl_gpt (MultiModalityCausalLM): The multimodal language model.
        tokenizer (Any): The tokenizer associated with the model.
        inputs (BatchedVLChatProcessorOutput): Processed inputs including image and text features.
    
    Returns:
        torch.Tensor: The generated token IDs representing the response.
    """
    try:
        # Ensure the model is in evaluation mode
        # vl_gpt.eval()

        # Run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.module.prepare_inputs_embeds(**inputs) \
            if args.ddp and "VL2" not in args.eval_model else \
            vl_gpt.prepare_inputs_embeds(**inputs)

        """
            Warning!!! DeepSeek-VL2 version and version 1.0 have modifications to the vl_gpt module, 
            vl_gpt.language_model.generate --> vl_gpt.language.generate
        """
        # Run the language model to generate a response
        outputs = vl_gpt.module.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        ) if args.ddp and "VL2" not in args.eval_model else \
        vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        # Decode the generated output into text
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
    
    except Exception as e:
        logging.error(f"Error in deepseekvl_forward: {e}")
        return None

def load_model_and_processor(args, model_path: str) -> Tuple[object, object, object]:
    """
    Loads the appropriate model and processor based on evaluation model type.
    
    Args:
        args: Configuration arguments containing model specifications
        model_path: Path to the pretrained model directory
        
    Returns:
        Tuple containing:
        - Processor for visual language chat tasks
        - Tokenizer for text processing
        - Loaded multi-modality language model
        
    Raises:
        ValueError: If CUDA is unavailable or model loading fails
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required but not available")

    # Load model components based on specified model type
    if args.eval_model == "DeepSeek-VL-7B-Chat":
        # Load components for DeepSeek-VL-7B-Chat variant
        from deepseek_vl.models.processing_vlm import BatchedVLChatProcessorOutput
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        # from deepseek_vl.utils.io import load_pil_images

        vl_chat_processor = VLChatProcessor.from_pretrained(
            model_path, 
            cache_dir=args.model_para_path
        )
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            cache_dir=args.model_para_path
        )
    else:
        # Load components for DeepSeek-VLv2 variant
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        # from deepseek_vl2.utils.io import load_pil_images 

        vl_chat_processor = DeepseekVLV2Processor.from_pretrained(
            model_path,
            cache_dir=args.model_para_path
        )
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            # device_map="auto",
            trust_remote_code=True,
            cache_dir=args.model_para_path
        )

    # Extract tokenizer from processor
    tokenizer = vl_chat_processor.tokenizer
    
    try:
        if args.eval_model == "DeepSeek-VL-7B-Chat":
            # Configure model for inference
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        else:
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()   # .cuda()
            if args.use_deepspeed:
                vl_gpt = deepspeed.init_inference(
                    vl_gpt,
                    tensor_parallel={"tp_size": 4},
                    dtype=torch.bfloat16,
                    # replace_with_kernel_inject=True
                    ).module

    except RuntimeError as e:
        raise ValueError(f"Failed to move model to CUDA: {str(e)}")

    return vl_chat_processor, tokenizer, vl_gpt

def run_model(args):
    """
    Run the DeepSeek-VL series model evaluation.
    
    Args:
        args: Argument parser containing configuration parameters.
    """
    model_path = args.module

    # Load model and processor
    vl_chat_processor, tokenizer, vl_gpt = load_model_and_processor(args, model_path)

    # Validate batch size
    if args.batchsize > 1:
        logging.error(
            "DeepSeek-VL-7B does not support batched processing of independent image-text pairs. Please set batchsize to 1."
            )
        raise ValueError("Invalid batch size: DeepSeek-VL-7B requires batchsize to be 1.")
    
    # logging.info(f"--------PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
    torch.cuda.empty_cache()

    # Initialize dataset and dataloader
    dataset = DeepSeekVL7BChatDataset(args, vl_chat_processor, vl_gpt)
    # Set sampler and DataLoader based on DDP mode
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if args.ddp else None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if not using DDP
        collate_fn=deepseek_vl_collate_fn
    )

    # Enable Distributed Data Parallel (DDP)
    with torch.no_grad():
        vl_gpt = torch.nn.parallel.DistributedDataParallel(vl_gpt, device_ids=[args.local_rank]) if args.ddp else vl_gpt

    
    start_time = time.time()

    # Initialize progress bar
    pbar = tqdm(total=len(data_loader), desc="Evaluating") if args.local_rank == 0 else None

    all_preds, all_labels = torch.empty(0), torch.empty(0)
    batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    for idx, batch_data in enumerate(data_loader):
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"GPU: {args.local_rank}")

        # Load batch data
        inputs_presence                     = batch_data.get('inputs_presence').to(vl_gpt.device)
        inputs_absence                      = batch_data.get('inputs_absence').to(vl_gpt.device)
        img_ids                             = batch_data.get('img_id')
        img_path                            = batch_data.get('img_path')
        categories                          = batch_data.get('category')
        presence_label                      = batch_data.get('presence_label')
        absence_label                       = batch_data.get('absence_label')
        presence_question_template          = batch_data.get('presence_question_template')
        absence_question_template           = batch_data.get('absence_question_template')

        # Model inference
        answer_presence = deepseekvl_forward(args, vl_gpt, tokenizer, inputs_presence)
        answer_absence = deepseekvl_forward(args, vl_gpt, tokenizer, inputs_absence)

        # Convert responses to binary
        response_presence_binary = convert_responses_to_binary([answer_presence])
        response_absence_binary = convert_responses_to_binary([answer_absence])

        # Gather results
        results = (img_ids, categories, response_presence_binary, presence_label, response_absence_binary, absence_label)

        if args.ddp:
            gathered_data = [None] * torch.distributed.get_world_size() if args.local_rank == 0 else None
            torch.distributed.gather_object(results, gathered_data, dst=0)
        else:
            gathered_data = [results]

        # Process results on rank 0
        if args.local_rank == 0 or not args.ddp:
            all_preds, all_labels, batch_preds, batch_labels = \
            extend_results(gathered_data, all_preds, all_labels, batch_preds, batch_labels)
            # Log intermediate metrics every 10 iterations
            if idx % 10 == 0 and idx > 1:
                accuracy, f1, precision, recall, mcc = metric_performances(batch_preds, batch_labels)
                logging.info(f"Batch {idx} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
                batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    end_time = time.time()
    formatted_time = format_elapsed_time(start_time, end_time)

    # Final evaluation metrics
    if args.local_rank == 0 or not args.ddp:
        accuracy, f1, precision, recall, mcc = metric_performances(all_preds, all_labels)
        logging.info(f"-Formatted time: {formatted_time}, GPUs num is {args.nproc_per_node}. The dataset has a total of {all_preds.size(0) if all_preds.size(0) == all_labels.size(0) else -1} samples complete evaluation of the performance of the metrics as follows: - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
        
        pred_tensor_dir = construct_tensor_path(label=False, args=args)
        label_tensor_dir = construct_tensor_path(label=True, args=args)

        # Save prediction and label tensors
        store_tensor(all_preds, pred_tensor_dir)
        store_tensor(all_labels, label_tensor_dir)

    if pbar is not None:
        pbar.close()
