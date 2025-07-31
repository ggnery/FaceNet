import torch
from torch import nn
from .inception_resnet_v2 import InceptionResNetV2
from .triplet_loss import TripletLoss
from typing import Optional, Tuple, Dict

class FaceNet(nn.Module):
    """
    FaceNet implementation using InceptionResNetV2 backbone and TripletLoss.
    Follows the original FaceNet paper specifications.
    """
    
    def __init__(self, device: torch.device, embedding_size: int = 512, 
                 pretrained_inception: Optional[str] = None):
        """
        Initialize FaceNet model.
        
        Args:
            device: Device to run the model on
            embedding_size: Size of face embeddings
            pretrained_inception: Path to pretrained InceptionResNetV2 weights
        """
        super(FaceNet, self).__init__()
        
        # Initialize InceptionResNetV2 backbone
        self.backbone = InceptionResNetV2(device, embedding_size=embedding_size)
        
        # Load pretrained weights if provided
        if pretrained_inception:
            self.backbone.load_state_dict(torch.load(pretrained_inception))
            
        # Initialize TripletLoss
        self.triplet_loss = TripletLoss(margin=0.2, embedding_size=embedding_size)
        
        self.device = device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network. Embeddings are NOT normalized"""
        return self.backbone(x)
    
    def compute_loss(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute triplet loss for a batch.
        
        Args:
            images: Batch of images (batch_size, 3, 299, 299)
            labels: Identity labels for each image
            
        Returns:
            Tuple of (loss, info_dict)
        """
        embeddings = self.forward(images)
        loss, info = self.triplet_loss(embeddings, labels)
        return loss, info
