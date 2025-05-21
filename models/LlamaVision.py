import torch
import time
from tqdm import tqdm
import csv
import logging
import os, sys
import copy
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any
from argparse import Namespace

import requests
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login

from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary,
    extend_results as extend_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor
    )

def load_model_and_processor(args: Namespace, model_path: str) -> tuple[MllamaForConditionalGeneration, AutoProcessor]:
    """
    Load a pre-trained Mllama model and its associated processor.

    Args:
        args (Namespace): Configuration arguments containing model parameter path.
        model_path (str): Path to the pre-trained model directory.

    Returns:
        tuple[MllamaForConditionalGeneration, AutoProcessor]:
            - Loaded Mllama model.
            - Corresponding processor for handling inputs.
    """
    # Authenticate Hugging Face account
    login(token='***********************')

    # Load the model with specified dtype and move it to GPU
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.model_para_path,
        ).cuda()
    
    # Load the processor for handling inputs
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=args.model_para_path)

    return model, processor

class Llama32VisionInstructDataset(Dataset):
    """
    A dataset class for vision-language instruction tuning with LLaVA-NeXT.

    This dataset loads image-category-label pairs from a CSV file and processes them 
    into a format suitable for multimodal inference with vision-language models.
    """
    def __init__(self, args: Any, model: torch.nn.Module, processor: Any) -> None:
        """
        Initializes the dataset with given arguments, model, and processor.

        Args:
            args (Any): Arguments containing dataset configurations.
            model (torch.nn.Module): The multimodal model used for processing.
            processor (Any): The processor for text and image preprocessing.
        """
        self.img_category_pairs: List[Tuple[str, str, int]] = []  # Stores (img_id, category, label) tuples
        self.datasets_img_path: str = args.DATASET_PATH.get(args.dataset)
        self.model = model
        self.processor = processor
        self.args = args

        # Load image-category-label pairs from the OOD dataset CSV file
        self._load_img_label_pairs(args.dataset_csv_path)
        
        # Template message format for inference
        self.message = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": None}
            ]}
        ]

    def _load_img_label_pairs(self, csv_file_path: str) -> None:
        """
        Loads image-category-label pairs from a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file containing image-category-label pairs.
        """
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                img_id, category, label = row[0], row[1], row[2]
                self.img_category_pairs.append((img_id, category, 1 if label.upper() == 'YES' else 0))

    def _process_data(self, question: str, img_path: str) -> Dict[str, torch.Tensor]:
        """
        Prepares the input data for the model by processing the image and text.

        Args:
            question (str): The question prompt.
            img_path (str): Path to the image file.

        Returns:
            Dict[str, torch.Tensor]: The processed inputs suitable for the model.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        messages = copy.deepcopy(self.message)
        messages[0].update({"content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]})
        
        # Load image
        image = Image.open(img_path)

        # Apply chat template to format input text
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenize and preprocess image-text pair
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        return inputs

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of image-category-label pairs.
        """
        return len(self.img_category_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a single sample from the dataset and processes it.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, Any]: A dictionary containing processed inputs and metadata.
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

        # Preparation for inference
        inputs_presence = self._process_data(presence_question_template, img_path)
        inputs_absence = self._process_data(absence_question_template, img_path)

        data = {
            'inputs_presence'               : inputs_presence, 
            'inputs_absence'                : inputs_absence, 
            'img_id'                        : img_id,
            'category'                      : category,
            'presence_label'                : presence_label,
            'absence_label'                 : absence_label,
            'presence_question_template'    : presence_question_template,
            'absence_question_template'     : absence_question_template
            }
        
        return data

def llama32_vision_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Custom collate function for batching vision data in LLama32 model.
    
    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary 
            contains the following keys:
            - 'inputs_presence': Presence-based input features
            - 'inputs_absence': Absence-based input features
            - 'img_id': Image identifiers
            - 'category': Category labels for each image
            - 'presence_label': Labels for presence-based questions
            - 'absence_label': Labels for absence-based questions
            - 'presence_question_template': Templates for presence-related questions
            - 'absence_question_template': Templates for absence-related questions

    Returns:
        Dict[str, List[Any]]: A dictionary containing batched data:
            - 'inputs_presence' (Any): Shared presence input features
            - 'inputs_absence' (Any): Shared absence input features
            - 'img_id' (List[Any]): List of image identifiers
            - 'category' (List[Any]): List of category labels
            - 'presence_label' (List[Any]): List of presence labels
            - 'absence_label' (List[Any]): List of absence labels
            - 'presence_question_template' (List[Any]): List of presence question templates
            - 'absence_question_template' (List[Any]): List of absence question templates
    """
    return {
        'inputs_presence'               : [item['inputs_presence'] for item in batch][0],
        'inputs_absence'                : [item['inputs_absence'] for item in batch][0],
        'img_id'                        : [item['img_id'] for item in batch],
        'category'                      : [item['category'] for item in batch],
        'presence_label'                : [item['presence_label'] for item in batch],
        'absence_label'                 : [item['absence_label'] for item in batch],
        'presence_question_template'    : [item['presence_question_template'] for item in batch],
        'absence_question_template'     : [item['absence_question_template'] for item in batch]
    }

def llama32_vision_forward(inputs: Dict[str, Any], model: Any, processor: Any) -> str:
    """
    Forward function for the LLama32 vision model to generate text responses.
    
    Args:
        inputs (Dict[str, Any]): Input dictionary containing preprocessed data for the model.
        model (Any): The LLama32 model instance, expected to support distributed execution.
        processor (Any): The processor instance used to decode the model's output.
    
    Returns:
        str: The final decoded text response after processing the model's output.
    """
    output = model.module.generate(**inputs, max_new_tokens=30)
    response = processor.decode(output[0]).split('\n')[-1].replace('<|eot_id|>', '')
    return response

def run_model(args: Any) -> None:
    """
    Runs the LLama3.2-11B-Vision-Instruct model for evaluation.
    
    Args:
        args (Any): Command-line arguments or configuration object containing required parameters.
    
    Raises:
        ValueError: If batch size is greater than 1, as it may degrade response quality.
    """
    model_path = args.module

    # Load model and processor
    model, processor = load_model_and_processor(args, model_path)

    # Validate batch size
    if args.batchsize > 1:
        logging.error(
            "Llama3.2-11B-Vision-Instruct might see a degradation in response quality with more images."
            )
        raise ValueError("Invalid batch size: Llama3.2-11B-Vision-Instruct requires batchsize to be 1.")

    torch.cuda.empty_cache()

    # Initialize dataset and dataloader
    dataset = Llama32VisionInstructDataset(args, model, processor)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, sampler=sampler, collate_fn=llama32_vision_collate_fn)

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
        inputs_presence                     = batch_data.get('inputs_presence').to(args.device)
        inputs_absence                      = batch_data.get('inputs_absence').to(args.device)
        img_ids                             = batch_data.get('img_id')
        img_path                            = batch_data.get('img_path')
        categories                          = batch_data.get('category')
        presence_label                      = batch_data.get('presence_label')
        absence_label                       = batch_data.get('absence_label')
        presence_question_template          = batch_data.get('presence_question_template')
        absence_question_template           = batch_data.get('absence_question_template')

        # Run inference
        response_presence = llama32_vision_forward(inputs_presence, model, processor)
        response_absence = llama32_vision_forward(inputs_absence, model, processor)

        # Convert responses to binary
        response_presence_binary = convert_responses_to_binary([response_presence])
        response_absence_binary = convert_responses_to_binary([response_absence])

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
