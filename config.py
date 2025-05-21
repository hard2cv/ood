import torch
import argparse
from scripts.utils import load_module_keys


DATASETS_CSV_PATH = {
    'coco': {
        'id': r'<your_csv_path>/coco_id.csv',
        'ood': r'<your_csv_path>/coco_ood.csv'
    },
    'nuscenes': {
        'id': r'<your_csv_path>/nuscenes_id.csv',
        'ood': r'<your_csv_path>/nuscenes_ood.csv'
    },
    'lvis': {
        'id': r'<your_csv_path>/lvis_id.csv',
        'ood': r'<your_csv_path>/lvis_ood.csv'
    },
    'cityscapes': {
        'id': r'<your_csv_path>/cityscapes_id.csv',
        'ood': r'<your_csv_path>/cityscapes_ood.csv'
    }
}

NUSCENES_CATEGORY_MAPPING = {
    "human.pedestrian.adult": "Adult Pedestrian",
    "human.pedestrian.child": "Child Pedestrian",
    "human.pedestrian.wheelchair": "Pedestrian in Wheelchair",
    "human.pedestrian.stroller": "Pedestrian with Stroller",
    "human.pedestrian.personal_mobility": "Pedestrian using Personal Mobility Device",  
    "human.pedestrian.police_officer": "Police Officer",
    "human.pedestrian.construction_worker": "Construction Worker",
    
    "animal": "Animal",
    
    "vehicle.car": "Car",
    "vehicle.motorcycle": "Motorcycle",
    "vehicle.bicycle": "Bicycle",
    "vehicle.bus.bendy": "Bendy Bus",
    "vehicle.bus.rigid": "Rigid Bus",
    "vehicle.truck": "Truck",
    "vehicle.construction": "Construction Vehicle",
    "vehicle.emergency.ambulance": "Ambulance",
    "vehicle.emergency.police": "Police Vehicle",
    "vehicle.trailer": "Trailer",
    
    "movable_object.barrier": "Barrier",
    "movable_object.trafficcone": "Traffic Cone",
    "movable_object.pushable_pullable": "Pushable/Pullable Object",
    # "movable_object.debris": "Debris",
    
    "static_object.bicycle_rack": "Bicycle Rack"
}


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Detection Model Parameters")
    
    # OOD Detector and Dataset parameters
    parser.add_argument(
        '--OOD_DETECTORS',
        type=dict,
        default={
            'clip': '<your_path>/CLIP_Para/ViT-L-14-336px.pt', 
            'blip2': '<your_path>/blip2_para',
            'groupvit': '<your_path>/MLLMPara',
            'lit': '<your_path>/MLLMPara'
        },
        help='OOD Detectors type'
        )
    parser.add_argument(
        '--DATASET_PATH',
        type=dict,
        default={
            'coco': '<your_dataset_dir>',
            'nuscenes': '<your_dataset_dir>',
            'lvis': '<your_dataset_dir>',
            'cityscapes': '<your_dataset_dir>'
            },
        help='all dataset path'
        )
    parser.add_argument(
        '--DATASET_LABEL',
        type=dict,
        default={
            'coco': r'<your_dataset_label_dir>',
            'lvis': r'<your_dataset_label_dir>'',
            'nuscenes': r'<your_dataset_label_dir>'',
            'cityscapes': r'<your_dataset_label_dir>'',
            },
        help='dataset path'
        )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='dataset path'
        )
    parser.add_argument(
        '--dataset_csv_path',
        type=str,
        default=None,
        help='dataset csv path'
        )
    parser.add_argument(
        '--model_para_path',
        type=str,
        default=None,
        help='model parameter path'
        )
    parser.add_argument(
        '--model_config',
        type=str,
        default=r'path/to/models/model_config.json',
        help='Path to the model configuration JSON file.'
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=r"<your_path>/evaluation_logs",
        help="Directory where logs will be stored. Default is './logs'."
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='coco', 
        choices=['coco', 'nuscenes', 'lvis', 'cityscapes'], 
        help='Currently selected dataset'
        )
    parser.add_argument(
        '--batchsize', 
        type=int, 
        default=1, 
        help='Batch size for ood detection. The number of samples processed together in one forward pass.'
        )
    parser.add_argument(
        '--detector_type',
        type=str, 
        default='', 
        choices=['clip', 'blip2', 'groupvit', 'lit'], 
        help='Selection of ood detectors to be used.'
        )
    parser.add_argument(
        '--logits_path', 
        type=str, 
        default=r'<your_path>', 
        help='Path to save the logits computed by the detector.'
        )
    parser.add_argument(
        '--ood_data_file_path', 
        type=str, 
        default=r'path/to/ood_data_file', 
        help='Paths for saving divided ood data.'
        )
    parser.add_argument(
        '--ood_data_num', 
        type=int, 
        default=0, 
        help="Record the number of data in the currently divided ood."
        )
    parser.add_argument(
        '--yes_ood_data_num', 
        type=int, 
        default=0, 
        help="Record the number of yes data in the currently divided ood."
        )
    parser.add_argument(
        '--no_ood_data_num', 
        type=int, 
        default=0, 
        help="Record the number of no data in the currently divided ood."
        )
    parser.add_argument(
        '--eval_model', 
        type=str, 
        default=None,
        choices=load_module_keys(r'path/to/models/model_config.json'), 
        help='The model to be used for evaluation. Choose from the available models based on the task requirements.'
    )
    parser.add_argument(
        "--enable_cot",
        action="store_true",
        help="Enable Chain-of-Thought (CoT) reasoning during testing. Default is False.",
    )
    parser.add_argument(
        "--is_ood",
        action="store_true",
        help="Specify whether testing on OOD data. Default is True (OOD data)."
    )
    parser.add_argument(
        "--presence_question_template", 
        type=str, 
        default="Does this image contain a [class]? (yes or no)", 
        help="Question template for checking the presence of a category in an image."
    )
    parser.add_argument(
        "--absence_question_template", 
        type=str, 
        default="Does this image not contain a [class]? (yes or no)", 
        help="Question template for checking the absence of a category in an image."
    )
    parser.add_argument(
        "--presence_cot_question_template", 
        type=str, 
        default="Does this image contain [class]? Yes or No? Let's break down the information step by step.", 
        help="COT-style question template for checking the presence of a category."
    )
    parser.add_argument(
        "--absence_cot_question_template", 
        type=str, 
        default="Does this image not contain [class]? Yes or No? Let's break down the information step by step.", 
        help="COT-style question template for checking the absence of a category."
    )
    parser.add_argument(
        "--module", 
        type=str,  
        help="Specify the module to use, like meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    # hyperparameter
    parser.add_argument(
        '--T', 
        type=float, 
        default=0.05, 
        help="The temperature of softmax (T); lower values increase confidence, higher values increase randomness."
        )

    # choice device
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        choices=['cuda', 'cpu'], 
        help='Device to run the model on'
        )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of processes per node (GPUs per node)."
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=10086,
        help="Master port for distributed training."
    )
    parser.add_argument(
        '--local_rank',
        type=int, 
        default=-1, 
        help='Local rank for distributed training'
        )
    parser.add_argument(
        '--local-rank',
        type=int, 
        default=-1, 
        help='Local rank for distributed training'
        )
    parser.add_argument(
        '--cuda_devices', 
        type=str, 
        default='0',
        help='Comma separated list of GPU ids to use, e.g., "0,1" for GPU 0 and 1. Default is "0".'
    )
    parser.add_argument(
        "--ddp", 
        action="store_true", 
        help="Enable Distributed Data Parallel (DDP)"
        )

    parser.add_argument(
        "--timestamp", 
        type=str, 
        default=None, 
        help="Current timestamp in format YYYY-MM-DD HH:MM (default: None)"
        )
    parser.add_argument(
        "--store_tensor_path",
        type=str,
        default=r'<your_path>',
        help='Specify the path to store tensors. If no other path is provided, the program will use this default path to store tensor data.'
    )

    return parser
