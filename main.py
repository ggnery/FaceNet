import torch
from model.inception_resnet_v2 import InceptionResNetV2

if __name__ == '__main__':
    device = torch.device("cpu")
    
    stem_model = InceptionResNetV2(device=device)
    stem_model.eval() # Set the model to evaluation mode
    
    input_tensor = torch.randn(1, 3, 299, 299) 
    print(f"Input tensor shape: {input_tensor.shape}")

    with torch.no_grad(): 
        output = stem_model(input_tensor)
        
    print(f"Output tensor shape: {output.shape}") 
