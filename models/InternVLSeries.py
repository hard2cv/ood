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
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary,
    extend_results as extend_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor
    )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_model_and_processor(args: Any, model_path: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer with specific configurations.

    Args:
        args (Any): Configuration object containing model parameters, including cache directory.
        model_path (str): Path to the pre-trained model.

    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]:
            - model: The loaded pre-trained model set to evaluation mode and moved to GPU.
            - tokenizer: The corresponding tokenizer for text processing.
    """
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced memory consumption.
        low_cpu_mem_usage=True,  # Optimize memory usage when loading the model.
        use_flash_attn=True,  # Enable FlashAttention for faster computation.
        trust_remote_code=True,  # Allow loading models with custom code.
        cache_dir=args.model_para_path  # Directory to store downloaded model weights.
    ).eval().cuda()  # Set model to evaluation mode and move it to GPU.

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,  # Allow tokenizer to load custom remote code.
        use_fast=False,  # Use the slow tokenizer for compatibility.
        cache_dir=args.model_para_path  # Directory to store downloaded tokenizer files.
    )

    return model, tokenizer

class InternVLSeriesDataset(Dataset):
    def __init__(self, args: Any) -> None:
        """
        Initialize the dataset with image-category pairs.
        
        Args:
            args (Any): Configuration object containing dataset parameters.
        """
        self.img_category_pairs = [] # Stores image-category pairs as (img_id, category, label)
        self.datasets_img_path = args.DATASET_PATH.get(args.dataset)  # Path to dataset images
        self.args = args
        self._load_img_label_pairs(args.dataset_csv_path)  # Load OOD image-category pairs

    def build_transform(self, input_size: int) -> T.Compose:
        """
        Builds a transformation pipeline for image preprocessing.

        Args:
            input_size (int): Target size for image resizing.
        
        Returns:
            torchvision.transforms.Compose: Transformation pipeline.
        """
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int) -> Tuple[int, int]:
        """
        Finds the closest aspect ratio from a given list.
        
        Args:
            aspect_ratio (float): Aspect ratio of the original image.
            target_ratios (List[Tuple[int, int]]): List of target aspect ratios.
            width (int): Original image width.
            height (int): Original image height.
            image_size (int): Target image size.
        
        Returns:
            Tuple[int, int]: Closest aspect ratio.
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> List[Image.Image]:
        """
        Dynamically preprocess an image by resizing and splitting it into blocks.
        
        Args:
            image (Image.Image): Input image.
            min_num (int, optional): Minimum number of blocks. Defaults to 1.
            max_num (int, optional): Maximum number of blocks. Defaults to 12.
            image_size (int, optional): Target image size. Defaults to 448.
            use_thumbnail (bool, optional): Whether to include a thumbnail. Defaults to False.
        
        Returns:
            List[Image.Image]: List of processed image blocks.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_file (str): Path to the image file.
            input_size (int, optional): Target image size. Defaults to 448.
            max_num (int, optional): Maximum number of processed images. Defaults to 12.
        
        Returns:
            torch.Tensor: Tensor of processed images.
        """
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
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

    def __len__(self) -> int:
        """
        Returns the total number of image-category pairs in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_category_pairs)

    def __getitem__(self, idx):
        img_id, category, presence_label = self.img_category_pairs[idx]
        absence_label = 1 - presence_label

        # Determine image path
        img_path = os.path.join(self.datasets_img_path, img_id)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = self.load_image(img_path)

        # Generate prompts
        presence_question_template = self.args.presence_question_template.replace('[class]', category)
        absence_question_template = self.args.absence_question_template.replace('[class]', category)

        messages_presence = f"<image>\n{presence_question_template}"
        messages_absence = f"<image>\n{absence_question_template}"

        data = {
            'img'                           : image, 
            'img_id'                        : img_id, 
            'category'                      : category,
            'presence_label'                : presence_label,
            'absence_label'                 : absence_label,
            'messages_presence'             : messages_presence, 
            'messages_absence'              : messages_absence,
            }
        return data

def InternVL_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batching samples in the InternVL dataset.

    Args:
        batch (List[Dict[str, Any]]): A list of samples, where each sample is a dictionary containing:
            - 'img' (torch.Tensor): Image tensor of shape (N, C, H, W), where N is the number of patches.
            - 'img_id' (str): Image identifier.
            - 'category' (str): Category associated with the image.
            - 'presence_label' (int): Presence label (1 if present, 0 if absent).
            - 'absence_label' (int): Absence label (1 if absent, 0 if present).
            - 'messages_presence' (str): Text message for presence query.
            - 'messages_absence' (str): Text message for absence query.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'images' (torch.Tensor): Concatenated image tensor along the batch dimension.
            - 'num_patches_list' (List[int]): List of number of patches per image in the batch.
            - 'img_id' (List[str]): List of image identifiers.
            - 'category' (List[str]): List of categories.
            - 'presence_label' (List[int]): List of presence labels.
            - 'absence_label' (List[int]): List of absence labels.
            - 'messages_presence' (List[str]): List of presence query messages.
            - 'messages_absence' (List[str]): List of absence query messages.
    """
    images = [item['img'] for item in batch]
    images = torch.cat(images, dim=0)       # Concatenate images along the batch dimension
    num_patches_list = [item['img'].size(0) for item in batch]  # Store number of patches per image
    
    # Collect metadata while preserving list structure
    metadata = {
        'img_id'                    : [item['img_id'] for item in batch],
        'category'                  : [item['category'] for item in batch],
        'presence_label'            : [item['presence_label'] for item in batch],
        'absence_label'             : [item['absence_label'] for item in batch],
        'messages_presence'         : [item['messages_presence'] for item in batch],
        'messages_absence'          : [item['messages_absence'] for item in batch]
    }
    return {
        'images': images,
        'num_patches_list': num_patches_list,
        **metadata
    }

def Internvl_forward(
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    imgs: torch.Tensor, 
    question: List[str], 
    num_patches_list: List[int], 
    generation_config: Dict[str, Any]
) -> List[str]:
    """
    Forward function for generating responses using the InternVL model.

    Args:
        model (PreTrainedModel): The pre-trained vision-language model.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
        imgs (torch.Tensor): Batched image tensor.
        question (List[str]): List of questions corresponding to each image.
        num_patches_list (List[int]): Number of patches per image in the batch.
        generation_config (Dict[str, Any]): Configuration dictionary for text generation.

    Returns:
        List[str]: Generated responses for each input question.
    """
    responses = model.module.batch_chat(
        tokenizer, 
        imgs,
        num_patches_list=num_patches_list,
        questions=question,
        generation_config=generation_config)
    return responses

def run_model(args: Any) -> None:
    """
    Execute the model evaluation with distributed data parallel (DDP) support.

    Args:
        args (Any): Arguments containing model path, batch size, local rank, and other configurations.

    Returns:
        None
    """
    model_path = args.module

    # Load model and processor
    model, tokenizer = load_model_and_processor(args, model_path)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # Initialize dataset and dataloader
    dataset = InternVLSeriesDataset(args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, sampler=sampler, collate_fn=InternVL_collate_fn)

    # Enable Distributed Data Parallel (DDP)
    model = model.to(f"cuda:{args.local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

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
        imgs                                = batch_data.get('images').to(torch.bfloat16).to(args.device)  
        img_ids                             = batch_data.get('img_id')
        categories                          = batch_data.get('category')
        presence_label                      = batch_data.get('presence_label')
        absence_label                       = batch_data.get('absence_label')
        messages_presence                   = batch_data.get('messages_presence')
        messages_absence                    = batch_data.get('messages_absence')
        num_patches_list                    = batch_data.get('num_patches_list')

        # Run inference
        responses_presence = Internvl_forward(model, tokenizer, imgs, messages_presence, num_patches_list, generation_config)
        responses_absence = Internvl_forward(model, tokenizer, imgs, messages_absence, num_patches_list, generation_config)

        # Convert responses to binary
        response_presence_binary = convert_responses_to_binary(responses_presence)
        response_absence_binary = convert_responses_to_binary(responses_absence)

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
