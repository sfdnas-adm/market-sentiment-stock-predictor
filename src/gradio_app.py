# gradio_app.py

import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# =======================
# 1) LSTM Model Definition
# =======================
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# =======================
# 2) Load the Model
# =======================
def load_model(model_path, input_size=5, hidden_size=128, num_layers=3, dropout=0.2):
    model = StockLSTM(input_size, hidden_size, num_layers, output_size=1, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# =======================
# 3) Prediction Function
# =======================
def predict_close_price(model, new_data_df, feature_cols, seq_length=20):
    if len(new_data_df) < seq_length:
        return "ERROR: Not enough rows. Need at least 20."

    # Scale
    scaler = MinMaxScaler()
    X = new_data_df[feature_cols].values
    X_scaled = scaler.fit_transform(X)

    # Take last seq_length rows
    X_input = X_scaled[-seq_length:]
    X_input = np.expand_dims(X_input, axis=0)
    X_tensor = torch.tensor(X_input, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_tensor).item()

    # Quick inverse transform hack
    dummy_array = np.zeros((1, len(feature_cols)))
    dummy_array[0, -1] = pred_scaled
    scaler.fit(X)  # re-fit on unscaled data
    pred_array = scaler.inverse_transform(dummy_array)
    predicted_close = pred_array[0, -1]
    return f"{predicted_close:.2f}"

# =======================
# 4) Gradio Interface
# =======================

# Adjust these to match your model
MODEL_PATH = "/content/nvidia_model.pth"
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
SEQ_LENGTH = 20

# Load once, reuse
model = load_model(MODEL_PATH, input_size=len(FEATURE_COLS), hidden_size=128, num_layers=3, dropout=0.2)

def predict_from_csv(csv_file):
    """
    This function reads a CSV file (uploaded by user)
    and returns the predicted close price.
    The CSV must have columns: Open, High, Low, Close, Volume
    and at least 20 rows.
    """
    df = pd.read_csv(csv_file.name)
    # Basic check
    for col in FEATURE_COLS:
        if col not in df.columns:
            return f"ERROR: CSV must contain column '{col}'."
    return predict_close_price(model, df, FEATURE_COLS, seq_length=SEQ_LENGTH)

# Simple Gradio interface
demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.components.File(label="Upload CSV with last 20 rows"),
    outputs="text",
    title="Stock Close Predictor",
    description="Upload a CSV with columns [Open, High, Low, Close, Volume]. Must have at least 20 rows."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
