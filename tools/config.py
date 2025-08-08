import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration class for FaceNet training parameters.
    
    This class loads configuration from a YAML file and provides easy access
    to all configuration parameters through attributes.
    """
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config_path = config_path
        self._load_config()
        self._set_attributes()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            self._config = yaml.safe_load(file)
    
    def _set_attributes(self):
        """Set configuration parameters as class attributes."""
        # Data Configuration
        data_config = self._config.get('data', {})
        self.data_root = data_config.get('root', "/data/vggface2")
        self.train_dir = data_config.get('train_dir', "train")
        self.val_dir = data_config.get('val_dir', "val")
        
        # Model Configuration
        model_config = self._config.get('model', {})
        self.embedding_size = model_config.get('embedding_size', 512)
        self.input_size = model_config.get('input_size', [299, 299])
        self.dropout_keep = model_config.get('dropout_keep', 0.8)
        
        loss_config = model_config.get('loss', {})
        self.margin = loss_config.get('margin', 0.2)
        
        # Training Configuration
        training_config = self._config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 2000)
        self.learning_rate = training_config.get('learning_rate', 0.05)
        self.faces_per_identity = training_config.get('faces_per_identity', 40)
        self.num_identities_per_batch = training_config.get('num_identities_per_batch', 45)
        self.ema_decay = training_config.get('ema_decay', 0.9999)
        self.ema_enabled = training_config.get('ema_enabled', True)
        self.weight_decay = training_config.get('weight_decay', 0.0001)
        
        # Learning rate schedule
        self.lr_schedule = training_config.get('lr_schedule', {0: self.learning_rate})
        
        # Calculated batch size
        self.batch_size = self.faces_per_identity * self.num_identities_per_batch
        
        # Checkpoint Configuration
        checkpoint_config = self._config.get('checkpoint', {})
        self.checkpoint_dir = checkpoint_config.get('dir', "./checkpoints")
        self.resume_checkpoint = checkpoint_config.get('resume', None)
        
        # Data Augmentation Configuration
        augmentation_config = self._config.get('augmentation', {})
        
        # Training augmentations
        train_aug = augmentation_config.get('train', {})
        self.random_horizontal_flip = train_aug.get('random_horizontal_flip', 0.5)
        self.random_rotation = train_aug.get('random_rotation', 10)
        
        color_jitter = train_aug.get('color_jitter', {})
        self.color_jitter_brightness = color_jitter.get('brightness', 0.1)
        self.color_jitter_contrast = color_jitter.get('contrast', 0.1)
        self.color_jitter_saturation = color_jitter.get('saturation', 0.1)
        
        # Normalization Configuration
        norm_config = self._config.get('normalization', {})
        self.norm_mean = norm_config.get('mean', [0.5, 0.5, 0.5])
        self.norm_std = norm_config.get('std', [0.5, 0.5, 0.5])
        
    
    def get_full_train_path(self) -> str:
        """Get full path to training directory."""
        return os.path.join(self.data_root, self.train_dir)
    
    def get_full_val_path(self) -> str:
        """Get full path to validation directory."""
        return os.path.join(self.data_root, self.val_dir)
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config.copy()
    
    def save_config(self, output_path: str):
        """Save current configuration to a new YAML file.
        
        Args:
            output_path (str): Path where to save the configuration
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"Config(embedding_size={self.embedding_size}, batch_size={self.batch_size}, epochs={self.num_epochs})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return f"Config(config_path='{self.config_path}', embedding_size={self.embedding_size})"
    
    def print_config(self):
        """Print all configuration parameters in a formatted way."""
        print("=" * 60)
        print("FaceNet Training Configuration")
        print("=" * 60)
        print()
        
        print("Data Configuration:")
        print(f"  Data root: {self.data_root}")
        print(f"  Training dir: {self.get_full_train_path()}")
        print(f"  Validation dir: {self.get_full_val_path()}")
        print()
        
        print("Model Configuration:")
        print(f"  Embedding size: {self.embedding_size}")
        print(f"  Input size: {self.input_size}")
        print(f"  Dropout probability: {self.dropout_keep}")
        print(f"  Loss:")
        print(f"    - Margin: {self.margin}")
        print()
        
        print("Training Configuration:")
        print(f"  Number of epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Learning rate schedule:")
        for epoch, lr in sorted(self.lr_schedule.items()):
            if lr == -1:
                print(f"    - Epoch {epoch}: End training")
            else:
                print(f"    - Epoch {epoch}: {lr}")
        print(f"  Faces per identity: {self.faces_per_identity}")
        print(f"  Identities per batch: {self.num_identities_per_batch}")
        print(f"  Total batch size: {self.batch_size}")
        print(f"  EMA decay: {self.ema_decay}")
        print(f"  EMA enabled: {self.ema_enabled}")
        print(f"  Weight decay: {self.weight_decay}")
        print()
        
        print("Checkpoint Configuration:")
        print(f"  Checkpoint directory: {self.checkpoint_dir}")
        print(f"  Resume checkpoint: {self.resume_checkpoint if self.resume_checkpoint else 'None (fresh start)'}")
        print()
        
        print("Data Augmentation (Training):")
        print(f"  Random horizontal flip: {self.random_horizontal_flip}")
        print(f"  Random rotation: {self.random_rotation}°")
        print(f"  Color jitter:")
        print(f"    - Brightness: {self.color_jitter_brightness}")
        print(f"    - Contrast: {self.color_jitter_contrast}")
        print(f"    - Saturation: {self.color_jitter_saturation}")
        print()
        
        print("Normalization:")
        print(f"  Mean: {self.norm_mean}")
        print(f"  Std: {self.norm_std}")
        print()
        
        print("=" * 60)