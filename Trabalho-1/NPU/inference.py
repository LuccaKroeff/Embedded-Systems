import torch
from torchvision import models, transforms
from PIL import Image
import time

USE_GPU = True

def get_device(use_gpu=True):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return preprocess(img).unsqueeze(0)  # Add batch dimension


def load_imagenet_labels():
    with open('classes.txt') as f:
        return [line.strip() for line in f.readlines()]


def run_inference(image_paths, use_gpu=True):
    device = get_device(use_gpu)
    print(f"Using device: {device}")

    # Load model
    model = models.inception_v3(weights='IMAGENET1K_V1')
    model.eval()
    model.to(device)
    #labels = load_imagenet_labels()
    #print("Loaded labbels")
    # Warmup run
    for img_path in image_paths:
        img_tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            model(img_tensor)

    total_time = 0
    for img_path in image_paths:
        try:
            img_tensor = preprocess_image(img_path).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            total_time += end_time - start_time
        
            #top_probs, top_idxs = torch.topk(probabilities, 3)

            # print(f"\nPredictions for {img_path}:")
            # for prob, idx in zip(top_probs, top_idxs):
            #     print(f"  {labels[idx]}: {prob.item():.2%}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    total_time *= 1000
    print(f"Total inference time: {total_time:.4f}ms")

# Example usage
if __name__ == '__main__':
    image_files = [
        'images/chairs.jpg',
        'images/notice_sign.jpg',
        'images/plastic_cup.jpg',
        'images/trash_bin.jpg',
    ]
    
    run_inference(image_files, use_gpu=USE_GPU)
