import torch
import torch.nn as nn
from typing import Optional
from copy import deepcopy


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Maintains a moving average of model parameters using:
    ema_param = decay * ema_param + (1 - decay) * model_param
    
    This provides a more stable model for evaluation and inference.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        """
        Initialize EMA.
        
        Args:
            model: The model to track with EMA
            decay: EMA decay rate (typically 0.999 or 0.9999)
            device: Device to store EMA parameters on
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create EMA model as a deep copy
        self.ema_model = deepcopy(model).to(self.device)
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        
        # Track updates
        self.num_updates = 0
        
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        
        Args:
            model: Current model to update EMA from
        """
        self.num_updates += 1
        
        # Adjust decay based on number of updates (warmup)
        # This helps with initial instability
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            # Get parameter iterators
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())
            
            # Update EMA parameters
            for name, ema_param in ema_params.items():
                if name in model_params:
                    model_param = model_params[name].to(self.device)
                    ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply EMA parameters to the model (temporarily).
        Use this before evaluation, then restore with restore().
        
        Args:
            model: Model to apply EMA parameters to
        """
        self._backup = {}
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        with torch.no_grad():
            for name, param in model_params.items():
                if name in ema_params:
                    # Backup original parameter
                    self._backup[name] = param.data.clone()
                    # Apply EMA parameter
                    param.data.copy_(ema_params[name].data.to(param.device))
    
    def restore(self, model: nn.Module):
        """
        Restore original model parameters after apply_shadow().
        
        Args:
            model: Model to restore parameters for
        """
        if not hasattr(self, '_backup'):
            return
            
        model_params = dict(model.named_parameters())
        
        with torch.no_grad():
            for name, param in model_params.items():
                if name in self._backup:
                    param.data.copy_(self._backup[name])
        
        del self._backup
    
    def state_dict(self):
        """Get EMA state dictionary for checkpointing."""
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'ema_model_state_dict': self.ema_model.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.ema_model.load_state_dict(state_dict['ema_model_state_dict'])
    
    def to(self, device):
        """Move EMA model to device."""
        self.device = device
        self.ema_model = self.ema_model.to(device)
        return self
    
    @property
    def module(self):
        """Access the EMA model."""
        return self.ema_model