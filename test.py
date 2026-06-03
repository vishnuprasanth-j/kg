 
import torch
 
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    print(f"Using GPU for training")
else:
    device = "cpu"
    print("No GPU detected, using CPU")
 