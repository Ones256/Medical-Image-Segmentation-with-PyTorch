import torch
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            fig, axes = plt.subplots(1, 3, figsize=(12,4))
            axes[0].imshow(images[0].cpu().squeeze(), cmap="gray")
            axes[0].set_title("Image")
            axes[1].imshow(masks[0].cpu().squeeze(), cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(outputs[0].cpu().squeeze() > 0.5, cmap="gray")
            axes[2].set_title("Prediction")
            plt.show()
            break
