import torch
import time
from tqdm import tqdm
import csv
import logging
import os, sys
import copy
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist
from typing import List, Dict, Tuple, Any

from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary,
    extend_merge_results as extend_merge_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor
    )

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


from torch.distributed._tensor import DTensor

def convert_dtensor_to_tensor(module: nn.Module) -> None:
    """
    Recursively converts all DTensor parameters and buffers in a module to regular Tensors.
    
    This function ensures that all DTensor objects are replaced with their local tensor copies,
    making the module compatible with standard PyTorch operations.
    
    Args:
        module (nn.Module): The PyTorch module whose DTensor parameters and buffers will be converted.
    
    Returns:
        None
    """
    # Convert DTensor parameters to standard Tensors
    for name, param in module.named_parameters(recurse=False):
        if isinstance(param, DTensor):
            local_tensor = param._local_tensor.detach().clone()
            new_param = nn.Parameter(local_tensor, requires_grad=param.requires_grad)
            module.register_parameter(name, new_param)
    
    # Convert DTensor buffers to standard Tensors
    for name, buffer in module.named_buffers(recurse=False):
        if isinstance(buffer, DTensor):
            local_tensor = buffer._local_tensor.detach().clone()
            module.register_buffer(name, local_tensor)
    
    # Recursively process child modules
    for child_module in module.children():
        convert_dtensor_to_tensor(child_module)

def qwen_vl_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Custom collate function for processing a batch of samples for Qwen-VL.
    
    This function organizes batch data into lists grouped by their corresponding keys.
    
    Args:
        batch (List[Dict[str, Any]]): A batch of samples where each sample is a dictionary
            containing various keys related to Qwen-VL inputs and labels.
    
    Returns:
        Dict[str, List[Any]]: A dictionary where each key corresponds to a list of values
            from the batch, ensuring structured batching for model input.
    """
    return {
        'messages_presence'             : [item['messages_presence'] for item in batch], 
        'messages_absence'              : [item['messages_absence'] for item in batch],
        'img_id'                        : [item['img_id'] for item in batch],
        'category'                      : [item['category'] for item in batch],
        'presence_label'                : [item['presence_label'] for item in batch],
        'absence_label'                 : [item['absence_label'] for item in batch],
        'presence_question_template'    : [item['presence_question_template'] for item in batch],
        'absence_question_template'     : [item['absence_question_template'] for item in batch]
    }


class QwenVLSeriesDataset(Dataset):
    """
    A PyTorch Dataset for processing Qwen-VL image-category pairs with presence and absence prompts.
    
    This dataset loads image-category-label pairs from a CSV file, generates corresponding presence and absence prompts,
    and structures the data for multimodal model input.
    """
    def __init__(self, args: Any) -> None:
        """
        Initializes the dataset with image-category pairs and paths.
        
        Args:
            args (Any): Configuration object containing dataset paths and templates.
        """
        self.img_category_pairs: List[Tuple[str, str, int]] = []  # Stores (img_id, category, label) tuples
        self.datasets_img_path: str = args.DATASET_PATH.get(args.dataset)
        self.args = args

        # Load image-category-label pairs from the OOD dataset CSV file
        self._load_img_label_pairs(args.dataset_csv_path)
        
        # Default message structure template
        self.message = [
            {"role": "user", "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": None}]}]

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

    def __len__(self) -> int:
        """
        Returns the total number of image-category pairs in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_category_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the sample at the specified index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Dict[str, Any]: A dictionary containing the processed sample with messages, labels, and prompts.
        """
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
        messages_presence = copy.deepcopy(self.message)
        messages_absence = copy.deepcopy(self.message)

        messages_presence[0].update({'content': [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": presence_question_template}]})
        messages_absence[0].update({'content': [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": absence_question_template}]}) 

        return {
            'messages_presence'             : messages_presence, 
            'messages_absence'              : messages_absence, 
            'img_id'                        : img_id,
            'category'                      : category,
            'presence_label'                : presence_label,
            'absence_label'                 : absence_label,
            'presence_question_template'    : presence_question_template,
            'absence_question_template'     : absence_question_template
        }

def load_model_and_processor(args: Any, model_path: str) -> Tuple[Any, AutoProcessor]:
    """
    Loads the specified Qwen model and its corresponding processor.
    
    Args:
        args (Any): Configuration object containing model evaluation settings and cache paths.
        model_path (str): Path to the pre-trained model.
    
    Returns:
        Tuple[Any, AutoProcessor]: The loaded model and processor.
    """
    if args.eval_model == "Qwen2-VL-7B-Instruct":
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            cache_dir=args.model_para_path).eval().to(args.device)
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=args.model_para_path)

    elif "Qwen2_5" in args.eval_model:
        from transformers import Qwen2_5_VLForConditionalGeneration
        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
            use_cache=False,    # Resolves flash_attn_2 issue
            cache_dir=args.model_para_path
        )

        # default processer
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_cache=False,     # Resolves flash_attn_2 issue
            cache_dir=args.model_para_path
            )
    return model, processor

def construct_batch_data(
    args: Any, 
    messages_presence: List[Any], 
    messages_absence: List[Any], 
    presence_label: List[int], 
    absence_label: List[int], 
    processor: Any
) -> Tuple[Any, List[int]]:
    """
    Constructs batch data for model inference.
    
    Args:
        args (Any): Configuration object containing batch size and model settings.
        messages_presence (List[Any]): List of presence-related messages.
        messages_absence (List[Any]): List of absence-related messages.
        presence_label (List[int]): List of presence labels.
        absence_label (List[int]): List of absence labels.
        processor (Any): Processor for text and vision data.
    
    Returns:
        Tuple[Any, List[int]]: Processed input tensors and corresponding labels.
    """
    messages = [elem for idx in range(args.batchsize) for elem in (messages_presence[idx], messages_absence[idx])]
    labels = [elem for idx in range(args.batchsize) for elem in (presence_label[idx], absence_label[idx])]

    # Prepare text inputs for batch inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    # Process vision-related inputs (images and videos)
    image_inputs, video_inputs = process_vision_info(messages)

    # Construct model inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        **({"padding_side": "left"} if "2_5" in args.eval_model else {}),
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    return inputs, labels

def qwenvl_forward(
    args: Any, 
    inputs: Any, 
    model: Any, 
    processor: Any
) -> List[str]:
    """
    Performs forward inference using the Qwen-VL model.
    
    Args:
        args (Any): Configuration object containing model settings.
        inputs (Any): Preprocessed input tensors for the model.
        model (Any): Qwen-VL model instance.
        processor (Any): Processor used for decoding model outputs.
    
    Returns:
        List[str]: List of decoded output texts.
    """
    # Batch Inference
    generated_ids = model.module.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample="2_5" not in args.eval_model)

    # Trim generated IDs based on input lengths
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode generated outputs into text
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts


def run_model(args):
    """
    Run the Qwen-VL Series model evaluation.
    
    Args:
        args: Argument parser containing configuration parameters.

    Returns:
        None
    """
    model_path = args.module

    # Load model and processor
    model, processor = load_model_and_processor(args, model_path)

    # Initialize dataset and dataloader
    dataset = QwenVLSeriesDataset(args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, sampler=sampler, collate_fn=qwen_vl_collate_fn)

    # Enable Distributed Data Parallel (DDP)
    # convert_dtensor_to_tensor(model)
    model = model.to(f"cuda:{args.local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    torch.cuda.empty_cache()
    start_time = time.time()

    # Initialize progress bar
    pbar = tqdm(total=len(data_loader), desc="Evaluating") if dist.get_rank() == 0 else None

    # Storage for predictions and labels
    all_preds, all_labels = torch.empty(0), torch.empty(0)
    batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    for idx, batch_data in enumerate(data_loader):
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"GPU: {dist.get_rank()}")

        # Load batch data
        messages_presence                   = batch_data.get('messages_presence')
        messages_absence                    = batch_data.get('messages_absence')
        img_ids                             = batch_data.get('img_id')
        img_path                            = batch_data.get('img_path')
        categories                          = batch_data.get('category')
        presence_label                      = batch_data.get('presence_label')
        absence_label                       = batch_data.get('absence_label')
        presence_question_template          = batch_data.get('presence_question_template')
        absence_question_template           = batch_data.get('absence_question_template')
        # print(presence_question_template)

        # Construct batch input
        inputs, labels = construct_batch_data(args, messages_presence, messages_absence, presence_label, absence_label, processor)

        # Run inference
        output_texts = qwenvl_forward(args, inputs, model, processor)

        # Convert model output to binary predictions
        response_binary = convert_responses_to_binary(output_texts)

        # Gather results across distributed processes
        results = (img_ids, categories, response_binary, labels)
        gathered_data = [None] * torch.distributed.get_world_size() if args.local_rank == 0 else None
        torch.distributed.gather_object(results, gathered_data, dst=0)

        # Process results on rank 0
        if dist.get_rank() == 0:
            all_preds, all_labels, batch_preds, batch_labels = \
            extend_merge_results(gathered_data, all_preds, all_labels, batch_preds, batch_labels)

            # Log intermediate metrics every 10 iterations
            if idx % 10 == 0 and idx > 1:
                accuracy, f1, precision, recall, mcc = metric_performances(batch_preds, batch_labels)
                logging.info(f"Batch {idx} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
                batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    end_time = time.time()
    formatted_time = format_elapsed_time(start_time, end_time)

    # Final evaluation metrics
    if dist.get_rank() == 0:
        accuracy, f1, precision, recall, mcc = metric_performances(all_preds, all_labels)
        logging.info(f"-Formatted time: {formatted_time}, GPUs num is {args.nproc_per_node}. The dataset has a total of {all_preds.size(0) if all_preds.size(0) == all_labels.size(0) else -1} samples complete evaluation of the performance of the metrics as follows: - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
        
        pred_tensor_dir = construct_tensor_path(label=False, args=args)
        label_tensor_dir = construct_tensor_path(label=True, args=args)

        # Save prediction and label tensors
        store_tensor(all_preds, pred_tensor_dir)
        store_tensor(all_labels, label_tensor_dir)

    if pbar is not None:
        pbar.close()