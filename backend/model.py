import torch
import torch.nn as nn
import numpy as np
import os

# Define the same model architecture
class StatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = StatModel()
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pth")
model.load_state_dict(torch.load(model_path))
model.eval()

# Dummy normalization values (update with your real scaler values later)
mean = np.array([25, 6, 6, 1, 1, 34])  # pts, reb, ast, stl, blk, min
std = np.array([5, 2, 2, 0.5, 0.5, 3])

# Predict over/under threshold
def predict_over_under(stats, threshold):
    """
    stats: list of [pts, reb, ast, stl, blk, min]
    threshold: float (e.g. 25 for points)
    """
    x = (np.array(stats) - mean) / std
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = model(x).item()
    prediction = "Yes" if prob > 0.5 else "No"
    confidence = f"{round(prob * 100, 1)}%" if prediction == "Yes" else f"{round((1 - prob) * 100, 1)}%"
    return {"prediction": prediction, "confidence": confidence}
