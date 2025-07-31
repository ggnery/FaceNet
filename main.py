import torch
import torch.optim as optim
from model.inception_resnet_v2 import InceptionResNetV2
from model.triplet_loss import TripletLoss, FaceNetBatchSampler
import numpy as np

def create_dummy_dataset(num_identities=100, samples_per_identity=50, image_size=(3, 299, 299)):
    """
    Create a dummy dataset for demonstration purposes.
    In practice, you would load your actual face dataset here.
    """
    total_samples = num_identities * samples_per_identity
    
    # Generate random images
    images = torch.randn(total_samples, *image_size)
    
    # Generate labels (identity IDs)
    labels = []
    for identity_id in range(num_identities):
        labels.extend([identity_id] * samples_per_identity)
    
    return images, torch.tensor(labels)

def train_facenet_example():
    """
    Example training loop demonstrating FaceNet triplet loss usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and move to device
    model = InceptionResNetV2(device=device, embedding_size=512)
    model.to(device)
    model.train()
    
    # Initialize triplet loss
    triplet_loss = TripletLoss(margin=0.2, embedding_size=512)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy dataset
    print("Creating dummy dataset...")
    images, labels = create_dummy_dataset(num_identities=50, samples_per_identity=40)
    
    # Create batch sampler following FaceNet paper guidelines
    batch_sampler = FaceNetBatchSampler(
        labels=labels.tolist(),
        batch_size=200,  # Smaller batch for demo (paper uses ~1800)
        samples_per_identity=4,  # Smaller for demo (paper uses ~40)
        shuffle=True
    )
    
    print("Starting training example...")
    
    # Training loop example
    for epoch in range(5):  # Just a few epochs for demo
        epoch_loss = 0.0
        epoch_stats = {
            'total_triplets': 0,
            'active_triplets': 0,
            'semi_hard_negatives': 0,
            'hard_negatives': 0
        }
        
        # Get one batch for this epoch (in real training, you'd iterate through all batches)
        batch_indices = next(iter(batch_sampler))
        
        # Get batch data
        batch_images = images[batch_indices].to(device)
        batch_labels = labels[batch_indices].to(device)
        
        print(f"\nEpoch {epoch + 1}")
        print(f"Batch size: {len(batch_indices)}")
        print(f"Unique identities in batch: {len(torch.unique(batch_labels))}")
        
        # Forward pass
        optimizer.zero_grad()
        embeddings = model(batch_images)
        
        # Compute triplet loss
        loss, mining_info = triplet_loss(embeddings, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        epoch_loss += loss.item()
        for key in epoch_stats:
            if key in mining_info:
                epoch_stats[key] += mining_info[key]
        
        # Print statistics
        print(f"Loss: {loss.item():.4f}")
        print(f"Triplets found: {mining_info.get('total_triplets', 0)}")
        print(f"Active triplets: {mining_info.get('active_triplets', 0)}")
        print(f"Semi-hard negatives: {mining_info.get('semi_hard_negatives', 0)}")
        print(f"Hard negatives: {mining_info.get('hard_negatives', 0)}")
        print(f"Avg positive distance: {mining_info.get('avg_pos_distance', 0):.4f}")
        print(f"Avg negative distance: {mining_info.get('avg_neg_distance', 0):.4f}")

def test_triplet_mining():
    """
    Test the triplet mining functionality with a controlled example.
    """
    print("\n" + "="*50)
    print("Testing Triplet Mining")
    print("="*50)
    
    device = torch.device("cpu")  # Use CPU for this test
    
    # Create a simple controlled dataset
    # 3 identities, each with 3 samples
    embeddings = torch.tensor([
        # Identity 0 - cluster around [1, 0]
        [1.0, 0.1], [1.1, -0.1], [0.9, 0.0],
        # Identity 1 - cluster around [-1, 0] 
        [-1.0, 0.1], [-1.1, -0.1], [-0.9, 0.0],
        # Identity 2 - cluster around [0, 1]
        [0.1, 1.0], [-0.1, 1.1], [0.0, 0.9]
    ], dtype=torch.float32)
    
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    # Initialize triplet loss
    triplet_loss = TripletLoss(margin=0.5, embedding_size=2)
    
    # Compute loss and mining statistics
    loss, mining_info = triplet_loss(embeddings, labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Mining statistics:")
    for key, value in mining_info.items():
        print(f"  {key}: {value}")

def evaluate_embeddings():
    """
    Example of how to evaluate the quality of learned embeddings.
    """
    print("\n" + "="*50)
    print("Evaluating Embeddings")
    print("="*50)
    
    device = torch.device("cpu")
    
    # Initialize model in evaluation mode
    model = InceptionResNetV2(device=device, embedding_size=512)
    model.eval()
    
    # Create sample images
    sample_images = torch.randn(6, 3, 299, 299)  # 6 sample images
    sample_labels = torch.tensor([0, 0, 1, 1, 2, 2])  # 3 identities, 2 samples each
    
    with torch.no_grad():
        embeddings = model(sample_images)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        print("Pairwise distances between embeddings:")
        print("(Same identity pairs should have smaller distances)")
        
        for i in range(len(sample_labels)):
            for j in range(i + 1, len(sample_labels)):
                same_identity = sample_labels[i] == sample_labels[j]
                distance = distances[i, j].item()
                status = "SAME" if same_identity else "DIFF"
                print(f"Sample {i} <-> Sample {j} ({status}): {distance:.4f}")

if __name__ == '__main__':
    print("FaceNet Triplet Loss Implementation Demo")
    print("=" * 50)
    
    # Test basic model functionality
    device = torch.device("cpu")
    model = InceptionResNetV2(device=device)
    model.eval()
    
    input_tensor = torch.randn(1, 3, 299, 299)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Output tensor shape: {output.shape}")
    
    # Test triplet mining with controlled data
    test_triplet_mining()
    
    # Demonstrate embedding evaluation
    evaluate_embeddings()
    
    # Run training example (commented out by default as it's computationally intensive)
    # train_facenet_example()
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("Uncomment train_facenet_example() to run the full training demo.") 
