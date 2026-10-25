"""
Practical 7: Positional Encoding and Layer Normalization
Name: Anjali Patalbansi | Program: MCA (AI & ML)

Install requirements:
pip install numpy matplotlib torch
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def positional_encoding(sequence_length, d_model):
    positions = np.arange(sequence_length)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(
        10000, (2 * (dimensions // 2)) / np.float32(d_model)
    )
    angles = positions * angle_rates
    encoding = np.zeros((sequence_length, d_model), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def main():
    print("PRACTICAL 7: POSITIONAL ENCODING AND LAYER NORMALIZATION")

    # Experiment 1: Positional encoding
    seq_len, d_model = 10, 16
    encoding = positional_encoding(seq_len, d_model)
    print("Positional encoding shape:", encoding.shape)
    print("First three positions (first eight dimensions):")
    print(np.round(encoding[:3, :8], 3))

    plt.figure(figsize=(10, 4))
    plt.imshow(encoding, aspect="auto")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token position")
    plt.title("Sinusoidal Positional Encoding")
    plt.colorbar(label="Encoding value")
    plt.tight_layout()
    plt.show()

    # Experiment 2: Layer normalization in a neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size=64, output_size=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.layer_norm = nn.LayerNorm(128)
            self.fc2 = nn.Linear(128, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = self.layer_norm(x)
            x = torch.relu(x)
            return self.fc2(x)

    torch.manual_seed(42)
    inputs = torch.randn(10, 64)
    model = SimpleNN()
    outputs = model(inputs)
    print("\nInput shape:", tuple(inputs.shape))
    print("Output shape:", tuple(outputs.shape))
    print("First sample output:", outputs[0].detach())

    print("\nResult: Positional encodings were generated and visualized.")
    print("Layer normalization was applied in a PyTorch neural network.")


if __name__ == "__main__":
    main()
