import time
from tqdm import tqdm
import csv
import logging
import os, sys
import base64
import torch
from typing import Tuple, List, Dict, Any

from scripts.utils import (
    convert_responses_to_binary as convert_responses_to_binary, 
    extend_results as extend_results,
    metric_performances as metric_performances,
    format_elapsed_time as format_elapsed_time,
    construct_tensor_path as construct_tensor_path,
    store_tensor as store_tensor,
    encode_image as encode_image
    )

from openai import OpenAI
import openai

model_key_api = "your_api_key"

def _load_img_label_pairs(csv_file_path: str) -> None:
    """
    Loads image-category-label pairs from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing image-category-label pairs.
    """
    img_category_pairs = []
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 3:
                continue  # Skip malformed rows
            img_id, category, label = row[0], row[1], row[2]
            img_category_pairs.append((img_id, category, 1 if label.upper() == 'YES' else 0))
    return img_category_pairs

def prepare_label(presence_label: int) -> Tuple[List[int], List[int]]:
    """
    Prepare presence and absence labels for binary classification.

    Args:
        presence_label (int): The presence label (0 or 1).

    Returns:
        Tuple[List[int], List[int]]: A tuple containing the presence and absence labels as lists.
    """
    absence_label = [1 - presence_label]
    presence_label = [presence_label]
    return presence_label, absence_label

def formalized_message(base64_image: str, question: str) -> List[Dict[str, Any]]:
    """
    Constructs a structured message payload containing a text question and an image in base64 format.
    
    Args:
        base64_image (str): The base64-encoded string of the image.
        question (str): The text-based question related to the image.
    
    Returns:
        List[Dict[str, Any]]: A structured message payload following a predefined format.
    
    Example:
        >>> message = formalized_message("/9j/4AAQSkZJRg...", "What is in this image?")
        >>> print(message)
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}}
            ]
        }]
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

def prepare_data(args: Any, img_id: str, category: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepares data for image-based question answering by constructing structured messages.
    
    Args:
        args (Any): The arguments object containing dataset path and prompt templates.
        img_id (str): The unique identifier of the image.
        category (str): The category name to be used in the prompts.
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two structured messages:
            - message_presence: A message asking about the presence of the category in the image.
            - message_absence: A message asking about the absence of the category in the image.
    
    Example:
        >>> args.dataset = "sample_dataset"
        >>> args.DATASET_PATH = {"sample_dataset": "/path/to/dataset"}
        >>> args.presence_question_template = "Is there a [class] in the image?"
        >>> args.absence_question_template = "Is there no [class] in the image?"
        >>> prepare_data(args, "image_001.jpg", "cat")
        ({"role": "user", "content": [...]}, {"role": "user", "content": [...]})
    """
    # Determine image path
    img_path = os.path.join(args.DATASET_PATH.get(args.dataset), img_id)

    # Generate prompts
    prompt_presence = args.presence_question_template.replace('[class]', category)
    prompt_absence = args.absence_question_template.replace('[class]', category)

    # Encode image to base64
    base64_image = encode_image(img_path)

    # Construct messages
    message_presence = formalized_message(base64_image, prompt_presence)
    message_absence = formalized_message(base64_image, prompt_absence)

    return message_presence, message_absence

def gpt_series_forward(client: Any, model_path: str, message: List[Dict[str, Any]]) -> str:
    """
    Sends a message to the GPT-4o model and retrieves the response.
    
    Args:
        client (Any): The API client used to communicate with the GPT model.
        model_path (str): The identifier or path of the GPT model to be used.
        message (List[Dict[str, Any]]): A structured message list in the required format.
    
    Returns:
        str: The text content of the response from the GPT model.
    
    Example:
        >>> client = SomeAPIClient()
        >>> model_path = "gpt-4o"
        >>> message = [{"role": "user", "content": "Hello, GPT!"}]
        >>> response = gpt_4o_forward(client, model_path, message)
        >>> print(response)
        "Hello! How can I assist you today?"
    """
    response = client.chat.completions.create(
        model=model_path,
        messages=message,
        timeout=60
        )
    response_content = response.choices[0].message.content
    return response_content

def run_model(args: Any) -> None:
    """
    Runs the GPT series model for image classification evaluation using a dataset.
    
    Args:
        args (Any): The argument object containing necessary configurations such as dataset path,
                    model details, and processing parameters.
    
    Returns:
        None: This function processes and evaluates predictions without returning values.
    
    Example:
        >>> args = Namespace(module="gpt-4o", dataset_csv_path="data.csv", nproc_per_node=4)
        >>> run_model(args)
    """
    model_path = args.module
    client = OpenAI(api_key=model_key_api)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )

    # logging model version
    logging.info(f"-Model Version: {response.model}")


    # Prepare data
    img_category_pair = _load_img_label_pairs(args.dataset_csv_path)

    start_time = time.time()

    # Storage for predictions and labels
    all_preds, all_labels = torch.empty(0), torch.empty(0)
    batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    for idx, data in enumerate(tqdm(img_category_pair)):
        # Load batch data
        img_ids, categories, presence_label = data
        presence_label, absence_label = prepare_label(presence_label)

        # Prepare messages for model
        message_presence, message_absence = prepare_data(args, img_ids, categories)

        try:
            response_content_presence = gpt_series_forward(client, model_path, message_presence)
            response_content_abesence = gpt_series_forward(client, model_path, message_absence)

        except Exception as e:
            logging.info(f'Error during request: {e} for image pair {data}')
            continue
        
        # Convert responses to binary
        response_presence_binary = convert_responses_to_binary([response_content_presence])
        response_absence_binary = convert_responses_to_binary([response_content_abesence])

        results = (img_ids, categories, response_presence_binary, presence_label, response_absence_binary, absence_label)
        formulated_data = [results]

        # Update prediction and label storage
        all_preds, all_labels, batch_preds, batch_labels = \
            extend_results(formulated_data, all_preds, all_labels, batch_preds, batch_labels)
        
        # Periodic evaluation
        if idx % 10 == 0 and idx > 1:
            accuracy, f1, precision, recall, mcc = metric_performances(batch_preds, batch_labels)
            logging.info(f"Batch {idx} - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
            batch_preds, batch_labels = torch.empty(0), torch.empty(0)

    end_time = time.time()
    formatted_time = format_elapsed_time(start_time, end_time)

    # Final evaluation metrics
    accuracy, f1, precision, recall, mcc = metric_performances(all_preds, all_labels)
    logging.info(f"-Formatted time: {formatted_time}, GPUs num is {args.nproc_per_node}. The dataset has a total of {all_preds.size(0) if all_preds.size(0) == all_labels.size(0) else -1} samples complete evaluation of the performance of the metrics as follows: - Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}, MCC: {mcc}")
    
    # Construct tensor file paths
    pred_tensor_dir = construct_tensor_path(label=False, args=args)
    label_tensor_dir = construct_tensor_path(label=True, args=args)

    # Save prediction and label tensors
    store_tensor(all_preds, pred_tensor_dir)
    store_tensor(all_labels, label_tensor_dir)
