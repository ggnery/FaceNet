import torch
from torch import nn
from .backbone.inception_resnet_v2 import InceptionResNetV2
from .loss.triplet_loss import TripletLoss
from typing import Optional, Tuple, Dict
import torch.nn.functional as F

class FaceNetInceptionResNetV2(nn.Module):
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
        super(FaceNetInceptionResNetV2, self).__init__()
        
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
        """Forward pass through the network. Embeddings are normalized"""
        
        # Normalize embeddings to unit sphere (L2 normalization)
        embeddings = self.backbone(x)
        return F.normalize(embeddings, p=2, dim=1)
    
    def compute_loss(self, images: torch.Tensor, labels: torch.Tensor, chunk_size) -> Tuple[torch.Tensor, Dict]:
        """
        Compute triplet loss for a batch using chunked forward pass to save memory.
        
        Args:
            images: Batch of images (batch_size, 3, 299, 299)
            labels: Identity labels for each image
            chunk_size: Size of chunks to process at once
            
        Returns:
            Tuple of (loss, info_dict)
        """
        batch_size = images.size(0)
        embeddings_list = []  
        
        # Process images in chunks
        for idx in range(0, batch_size, chunk_size):  
            chunk_images = images[idx:idx+chunk_size]
            embeddings_chunk = self.forward(chunk_images)
            embeddings_list.append(embeddings_chunk)
        
        embeddings = torch.cat(embeddings_list, dim=0)
        
        loss, info = self.triplet_loss(embeddings, labels)
        return loss, info
