from PIL import Image
import requests
import copy
import csv
import torch
import os, sys
import logging
import time
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List

from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary,
    extend_results as extend_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor
    )

from transformers import PreTrainedTokenizer, PreTrainedModel

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


def load_model_and_processor(
    args: object, model_path: str
) -> Tuple[PreTrainedTokenizer, PreTrainedModel, object, int]:
    """
    Load a pre-trained model, tokenizer, and image processor.

    Args:
        args (object): Argument object containing necessary configurations.
        model_path (str): Path to the pre-trained model directory.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel, object, int]:
            - tokenizer: The tokenizer for text processing.
            - model: The pre-trained model instance.
            - image_processor: The image processor for handling images.
            - max_length: Maximum sequence length for tokenized inputs.
    """
    model_name = "llava_llama3"

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path, 
        None, 
        model_name, 
        device_map=None, 
        cache_dir=args.model_para_path) # Add any other thing you want to pass in llava_model_args
    return tokenizer, model, image_processor, max_length

class LLaVANeXTDataset(Dataset):
    """
    Custom dataset for LLaVA-NeXT, handling image-category-label pairs.
    """
    def __init__(
        self, args: object, tokenizer: object, model: object, image_processor: object
    ) -> None:
        """
        Initialize the dataset.

        Args:
            args (object): Configuration arguments.
            tokenizer (object): Tokenizer for text processing.
            model (object): Pre-trained model instance.
            image_processor (object): Image processor for handling images.
        """
        self.img_category_pairs = []    # Stores image-category pairs as (img_id, category, label)
        self.datasets_img_path = args.DATASET_PATH.get(args.dataset)  # Dataset path
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        self._load_img_label_pairs(args.dataset_csv_path)  # Load image-category-label pairs
        
    def _load_img_label_pairs(self, csv_file_path: str) -> None:
        """
        Load image-category-label pairs from a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file.
        """
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                img_id, category, label = row[0], row[1], row[2]
                self.img_category_pairs.append((img_id, category, 1 if label.upper() == 'YES' else 0))

    def _prompt_process(self, question_template: str) -> torch.Tensor:
        """
        Process prompt for the model.

        Args:
            question_template (str): Template for the prompt.

        Returns:
            torch.Tensor: Tokenized prompt tensor.
        """
        conv_template = "llava_llama_3"     # Ensure the correct chat template for different models
        question = DEFAULT_IMAGE_TOKEN + f"\n{question_template}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        conv.tokenizer = self.tokenizer
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.args.device)
        return input_ids

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.img_category_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: Dictionary containing input tensors and metadata.
        """
        img_id, category, presence_label = self.img_category_pairs[idx]
        absence_label = 1 - presence_label
        
        # Determine image path
        img_path = os.path.join(self.datasets_img_path, img_id)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        
        # Generate prompts
        presence_question_template = self.args.presence_question_template.replace('[class]', category)
        absence_question_template = self.args.absence_question_template.replace('[class]', category)

        image = Image.open(img_path)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.args.device) for _image in image_tensor]

        # Process texts
        input_ids_presence = self._prompt_process(presence_question_template)
        input_ids_absence = self._prompt_process(absence_question_template)
        
        image_sizes = [image.size]

        data = {
            'input_ids_presence'                    : input_ids_presence, 
            'input_ids_absence'                     : input_ids_absence, 
            'image_tensor'                          : image_tensor, 
            'image_sizes'                           : image_sizes,
            'img_path'                              : img_path,
            'img_id'                                : img_id, 
            'category'                              : category, 
            'presence_label'                        : presence_label, 
            'absence_label'                         : absence_label, 
            'presence_question_template'            : presence_question_template, 
            'absence_question_template'             : absence_question_template
            }
        return data

def LLaVANeXT_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching LLaVA-NeXT dataset samples.

    Args:
        batch (List[Dict[str, Any]]): List of samples from the dataset.

    Returns:
        Dict[str, Any]: Batched data with input tensors and metadata.
    """
    input_ids_presence = [item['input_ids_presence'] for item in batch][0]
    input_ids_absence = [item['input_ids_absence'] for item in batch][0]
    image_tensor = [item['image_tensor'] for item in batch][0]
    image_sizes = [item['image_sizes'] for item in batch]
    img_path = [item['img_path'] for item in batch]
    img_id = [item['img_id'] for item in batch]
    category = [item['category'] for item in batch]
    presence_label = [item['presence_label'] for item in batch]
    absence_label = [item['absence_label'] for item in batch]
    presence_question_template = [item['presence_question_template'] for item in batch][0]
    absence_question_template = [item['absence_question_template'] for item in batch][0]

    return {
        'input_ids_presence'                    : input_ids_presence, 
        'input_ids_absence'                     : input_ids_absence, 
        'image_tensor'                          : image_tensor, 
        'image_sizes'                           : image_sizes,
        'img_path'                              : img_path,
        'img_id'                                : img_id,
        'category'                              : category,
        'presence_label'                        : presence_label,
        'absence_label'                         : absence_label,
        'presence_question_template'            : presence_question_template,
        'absence_question_template'             : absence_question_template
        }

def llava_next_forward(
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    image_sizes: List[Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer
) -> List[str]:
    """
    Perform forward inference with the LLaVA-Next model.

    Args:
        input_ids (torch.Tensor): Tokenized input text tensor.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        image_sizes (List[Any]): List of image sizes.
        model (PreTrainedModel): The LLaVA-Next model.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding output.

    Returns:
        List[str]: Decoded text outputs from the model.
    """
    cont = model.module.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs

def run_model(args: Any) -> None:
    """
    Run the LLaVA-NeXT model for inference and evaluation in a distributed setting.

    Args:
        args (Any): Configuration object containing dataset, model, and training parameters.

    Raises:
        ValueError: If batch size is greater than 1, since LLaVA-NeXT does not support batch processing.
    """
    model_path = args.module

    # Load model and processor
    tokenizer, model, image_processor, _ = load_model_and_processor(args, model_path)

    # Validate batch size
    if args.batchsize > 1:
        logging.error(
            "LLaVA-NeXT does not support batched processing of independent image-text pairs. Please set batchsize to 1."
            )
        raise ValueError("Invalid batch size: LLaVA-NeXT requires batchsize to be 1.")

    # Initialize dataset and dataloader
    dataset = LLaVANeXTDataset(args, tokenizer, model, image_processor)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batchsize, 
        sampler=sampler, 
        collate_fn=LLaVANeXT_collate_fn
        )
    
    # Enable Distributed Data Parallel (DDP)
    with torch.no_grad():
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
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
        input_ids_presence                  = batch_data.get('input_ids_presence').to(args.device)
        input_ids_absence                   = batch_data.get('input_ids_absence').to(args.device)
        image_tensor                        = batch_data.get('image_tensor')[0].to(args.device)
        image_sizes                         = batch_data.get('image_sizes')
        img_ids                             = batch_data.get('img_id')
        img_path                            = batch_data.get('img_path')
        categories                          = batch_data.get('category')
        presence_label                      = batch_data.get('presence_label')
        absence_label                       = batch_data.get('absence_label')
        presence_question_template          = batch_data.get('presence_question_template')
        absence_question_template           = batch_data.get('absence_question_template')

        # Model inference
        text_outputs_presence = llava_next_forward(input_ids_presence, image_tensor, image_sizes, model, tokenizer)
        text_outputs_absence = llava_next_forward(input_ids_absence, image_tensor, image_sizes, model, tokenizer)

        # Convert responses to binary
        response_presence_binary = convert_responses_to_binary(text_outputs_presence)
        response_absence_binary = convert_responses_to_binary(text_outputs_absence)

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