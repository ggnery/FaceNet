import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from collections import defaultdict


class TripletLoss(nn.Module):
    """
    FaceNet Triplet Loss implementation following the original paper.
    
    This implementation includes:
    1. Online triplet generation with hard positive/negative selection
    2. Semi-hard negative mining
    3. All anchor-positive pairs with hard negative selection
    4. Proper batch construction with identity sampling
    """
    
    def __init__(self, margin: float = 0.2, embedding_size: int = 512):
        """
        Initialize TripletLoss.
        
        Args:
            margin: The margin alpha for triplet loss
            embedding_size: Size of the embedding vectors
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.embedding_size = embedding_size
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute triplet loss with online triplet mining.
        
        Args:
            embeddings: Tensor of shape (batch_size, embedding_size)
            labels: Tensor of shape (batch_size,) containing identity labels
            
        Returns:
            Tuple of (loss, info_dict) where info_dict contains mining statistics
        """
        
        
        triplets, mining_info = self.mine_triplets(embeddings, labels) # Online mining
        
        if len(triplets) == 0:
            # No valid triplets found, return zero loss
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), mining_info
        
        # Extract anchor, positive, negative indices
        anchor_indices, positive_indices, negative_indices = zip(*triplets)
        
        # Get embeddings for anchors, positives, negatives
        anchor_embeddings = embeddings[list(anchor_indices)]
        positive_embeddings = embeddings[list(positive_indices)]
        negative_embeddings = embeddings[list(negative_indices)]
        
        # Compute distances
        pos_distances = torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1)
        neg_distances = torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1)
        
        # Triplet loss
        losses = F.relu(pos_distances - neg_distances + self.margin)
        loss = torch.mean(losses)
        
        # Update mining info
        mining_info.update({
            'triplet_loss': loss.item(),
            'avg_pos_distance': torch.mean(pos_distances).item(),
            'avg_neg_distance': torch.mean(neg_distances).item(),
            'active_triplets': torch.sum(losses > 0).item(),
            'total_triplets': len(triplets)
        })
        
        return loss, mining_info
    
    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[List, dict]:
        """
        Mine triplets using the FaceNet strategy:
        - Use all anchor-positive pairs
        - Select semi-hard negatives
        
        Args:
            embeddings: Normalized embeddings
            labels: Identity labels
            
        Returns:
            Tuple of (triplets, mining_info)
        """        
        # Compute pairwise squared distances
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        
        # Group indices by label
        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label.item()].append(i)
        
        triplets = []
        mining_stats = {
            'total_pairs': 0,
            'valid_pairs': 0,
            'semi_hard_negatives': 0,
            'hard_negatives': 0,
            'easy_negatives': 0
        }
        
        # For each identity with multiple samples
        for label, indices in label_to_indices.items():
            if len(indices) < 2:
                continue  # Need at least 2 samples for positive pairs
                
            # Generate all anchor-positive pairs for this identity
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    anchor_idx = indices[i]
                    positive_idx = indices[j]
                    
                    mining_stats['total_pairs'] += 1
                    
                    # Get anchor-positive distance
                    ap_distance = distances[anchor_idx, positive_idx]
                    
                    # Find semi-hard negatives
                    negative_idx = self.find_semi_hard_negative(
                        anchor_idx, ap_distance, distances, labels
                    )
                    
                    if negative_idx is not None:
                        triplets.append((anchor_idx, positive_idx, negative_idx))
                        mining_stats['valid_pairs'] += 1
                        
                        # Classify negative type
                        an_distance = distances[anchor_idx, negative_idx]
                        if an_distance > ap_distance and an_distance < ap_distance + self.margin:
                            mining_stats['semi_hard_negatives'] += 1
                        elif an_distance <= ap_distance:
                            mining_stats['hard_negatives'] += 1
                        else:
                            mining_stats['easy_negatives'] += 1
        
        return triplets, mining_stats
    
    def find_semi_hard_negative(self, anchor_idx: int, ap_distance: torch.Tensor, distances: torch.Tensor, labels: torch.Tensor) -> Optional[int]:
        """
        Find a semi-hard negative for the given anchor-positive pair.
        
        Semi-hard negative: d(a,n) > d(a,p) and d(a,n) < d(a,p) + margin
        
        Args:
            anchor_idx: Index of anchor
            ap_distance: Distance between anchor and positive
            distances: Pairwise distance matrix
            labels: Identity labels
            
        Returns:
            Index of semi-hard negative or None if not found
        """
        anchor_label = labels[anchor_idx]
        
        # Get distances from anchor to all other samples
        anchor_distances = distances[anchor_idx]
        
        # Find candidates (different identity from anchor)
        different_identity_mask = (labels != anchor_label)
        candidate_indices = torch.where(different_identity_mask)[0]
        
        if len(candidate_indices) == 0:
            return None
        
        # Semi-hard condition: d(a,n) > d(a,p) and d(a,n) < d(a,p) + margin
        candidate_distances = anchor_distances[candidate_indices]
        semi_hard_mask = (candidate_distances > ap_distance) & (candidate_distances < ap_distance + self.margin)
        
        semi_hard_candidates = candidate_indices[semi_hard_mask]
        
        if len(semi_hard_candidates) > 0:
            # Randomly select one semi-hard negative
            return semi_hard_candidates[torch.randint(0, len(semi_hard_candidates), (1,))].item()
        
        # If no semi-hard negatives, fall back to hardest negative
        # (closest negative that's still different identity)
        if len(candidate_indices) > 0:
            hardest_idx = torch.argmin(candidate_distances)
            return candidate_indices[hardest_idx].item()
        
        return None
    
