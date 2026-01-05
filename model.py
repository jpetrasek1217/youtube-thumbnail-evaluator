import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from hybrid_nn import HybridEvaluator
from huggingface_hub import hf_hub_download


def load_model(model_path: str = None) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = hf_hub_download(
        repo_id="josephpetrasek/youtube-video-evaluator",
        filename="model.pth"
    )
    
    model = HybridEvaluator(num_numeric_features=6, num_classes=8, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def parse_video_length(video_length: str) -> float:
    """
    Convert video length to seconds. Supports:
    - MM:SS (e.g., "10:30")
    - HH:MM:SS (e.g., "01:10:30")
    - DD:HH:MM:SS (e.g., "01:01:10:30")
    
    Args:
        video_length: String in format MM:SS, HH:MM:SS, or DD:HH:MM:SS
    
    Returns:
        Total seconds as float
    """
    try:
        parts = video_length.split(':')
        
        if len(parts) == 2:
            # MM:SS
            minutes, seconds = map(int, parts)
            total_seconds = minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 4:
            # DD:HH:MM:SS
            days, hours, minutes, seconds = map(int, parts)
            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid format. Expected MM:SS, HH:MM:SS, or DD:HH:MM:SS, got {video_length}")
        
        return float(total_seconds)
    except Exception as e:
        print(f"Error parsing video length: {e}")
        return 0.0

def normalize_input(value: float, min_val: float, max_val: float, log_scale=False) -> float:
    """
    Normalize value to [0, 1] range
    """
    if log_scale:
        value = np.log1p(value)
        min_val = np.log1p(min_val)
        max_val = np.log1p(max_val)
    return float((value - min_val) / (max_val - min_val + 1e-9))

def preprocess_input(input_data: Dict) -> torch.Tensor:
    """
    Preprocess input data into tensor format suitable for model.
    
    IMPORTANT: Adjust this based on your model's expected input format.
    
    Args:
        input_data: Dictionary containing user input
    
    Returns:
        torch.Tensor ready for model inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract and normalize numerical features
    video_length_seconds = parse_video_length(input_data["videoLength"])
    
    # Normalize features (adjust min/max values based on your data)
    features = [
        normalize_input(video_length_seconds, 0, 134675, log_scale=True),  # longest video in dataset
        normalize_input(input_data["videoTitleLength"], 1, 100), # need to change these so they are not hardcoded in v2
        normalize_input(input_data["channelSubscribers"], 0, 453000000, log_scale=True),
        normalize_input(input_data["totalChannelViews"], 0, 326000000000, log_scale=True),
        normalize_input(input_data["totalVideos"], 0, 645000),
        normalize_input(input_data["channelAgeYears"], 0, 20),
    ]
    
    # Convert to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    return features_tensor

def predict(model: nn.Module, input_data: Dict) -> float:
    """
    Make a prediction using the loaded model.
    
    Args:
        model: The loaded PyTorch model
        input_data: Dictionary with user input
    
    Returns:
        Prediction value (CTR prediction or probability)
    """
    with torch.no_grad():
        # Preprocess input
        input_tensor = preprocess_input(input_data)
        # Get prediction
        output = model(
            input_data["thumbnail"],
            input_data["videoTitle"], 
            input_tensor,
            torch.tensor(input_data["videoUploadHour"], dtype=torch.long).unsqueeze(0), 
            torch.tensor(input_data["videoUploadDayofWeek"], dtype=torch.long).unsqueeze(0)
        )
        # Convert to Python float
        prediction = torch.argmax(output, dim=1).item()

    return prediction
