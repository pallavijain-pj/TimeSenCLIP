import numpy as np
import torch
import torch.nn.functional as F

class TimeSenCLIPPreprocessor:
    """
    Preprocess satellite images for TimeSenCLIP.
    Supports [T, C, H, W] format input.
    """

    def __init__(self,  device='cpu'):#mean, std,
        
        mean = [67.0, 122.0, 93.27, 158.5, 160.77, 174.27, 162.27, 149.0, 84.5, 66.27]
        std =  [2089.0, 2598.45, 3214.5, 3620.45, 4033.61, 4613.0, 4825.45, 4945.72, 5140.84, 4414.45]
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).float().to(device)  # [1, C, 1, 1]
        self.std = torch.tensor(std).view(1, -1, 1, 1).float().to(device)    # [1, C, 1, 1]
        self.device = device

    def preprocess(self, image_tensor):
        """
        Args:
            image_tensor (torch.Tensor): shape [T, C, H, W], raw values

        Returns:
            torch.Tensor: shape [1, T, C, H, W], normalized
        """
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor).float()

        image_tensor = image_tensor.to(self.device)

        # Normalize each time step independently
        image_tensor = (image_tensor - self.mean) / (self.std + 1e-6)  # [T, C, H, W]
        image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        image_tensor = image_tensor.unsqueeze(0)  # [1, T, C, H, W] for batch dimension
        
        return image_tensor