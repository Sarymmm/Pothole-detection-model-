import torch
import torchvision.models as models
import torch.nn as nn

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    model = model.to(device)

    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "training", "model.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device