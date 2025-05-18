import torch
from torchvision import transforms
from PIL import Image
from .digitCNN import SmallDigitCNN
import os

# Load model
model = SmallDigitCNN()
base_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(base_dir, "digit_model_weights_new.pth")

model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Prediction function
def predict_digit(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # [1, 1, 50, 50]
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example usage
# for i in range(10):
#     img_path = f"digits/{str(i)}.png"
#     pred = predict_digit(img_path)
#     print(f"Correct: {i} Pred: {pred}")
#

transform1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def predict_cell(cell_img_np):
    # cell_img_np: 2D numpy array from OpenCV (grayscale)

    # Convert to PIL, apply transform
    pil_img = Image.fromarray(cell_img_np)
    tensor = transform1(pil_img).unsqueeze(0)  # Shape: [1, 1, 50, 50]

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)

    return pred.item(), conf.item()




