import torch
from data_loader import get_dataloader
from model import SimpleCNN
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    image_dir = "/content/drive/MyDrive/PROJECT_F/2d_images"
    mask_dir = "/content/drive/MyDrive/PROJECT_F/2d_masks"

    dataloader = get_dataloader(image_dir, mask_dir, batch_size=4)

    model = SimpleCNN()

    train_model(model, dataloader, epochs=5, lr=0.001, device="cuda" if torch.cuda.is_available() else "cpu")
    evaluate_model(model, dataloader)
