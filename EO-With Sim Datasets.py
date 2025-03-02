import os
import time
import json
import struct
import zlib
import hmac
import hashlib
import logging
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ------------------------------
# Global Configuration (modifiable; could be loaded from file)
# ------------------------------
CONFIG = {
    "data": {
        "image_shape": (256, 256),
        "num_classes": 10,
        "num_samples": 500  # samples per individual dataset
    },
    "model": {
        "scale_factor": 2
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 32,
        "learning_rate": 1e-3
    },
    "rf": {
        "modulation": "QPSK",
        "band_speed": 1e6,  # bits per second
        "redundancy_factor": 2
    },
    "security": {
        "encryption_key": None,  # generated if not provided
        "hmac_key": None
    },
    "scheduler": {
        "max_workers": 4  # For multithreading (if needed)
    }
}

# ------------------------------
# Logging Configuration for Traceability
# ------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------
# Data Ingestion and Preprocessing (Synchronous)
# ------------------------------
def load_data(image_shape):
    """Simulate loading a satellite image."""
    data = np.random.rand(*image_shape).astype(np.float32)
    logger.info("Data loaded (simulated).")
    return data

def preprocess_data(data):
    """Normalize data to [0, 1] range."""
    data_min, data_max = data.min(), data.max()
    preprocessed = (data - data_min) / (data_max - data_min) if data_max > data_min else data
    logger.info("Data preprocessed (normalized).")
    return preprocessed

# ------------------------------
# Dataset for Simulation (Used for Local Training)
# ------------------------------
class SimulatedImageDataset(Dataset):
    def __init__(self, num_samples, image_shape, num_classes, dataset_id=0):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.dataset_id = dataset_id  # identifier for the dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In a real scenario, replace this with your actual data loader.
        image = preprocess_data(np.random.rand(*self.image_shape).astype(np.float32))
        # Optionally, you can embed the dataset_id into the data for debugging.
        label = np.random.randint(0, self.num_classes)
        return torch.from_numpy(image), label

# ------------------------------
# Model Architecture: Advanced Feature Extractor + Classifier
# ------------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn_output = attn_weights @ V
        return attn_output

class AdvancedFeatureExtractor(nn.Module):
    def __init__(self, input_dim, scale_factor):
        super(AdvancedFeatureExtractor, self).__init__()
        self.fc_reduce = nn.Linear(input_dim, 256)
        self.num_tokens = 16      # Divide reduced features into 16 tokens
        self.token_dim = 16       # Each token dimension (16*16=256)
        self.attention = SelfAttention(embed_dim=self.token_dim)
        self.fc_out = nn.Linear(512, 64)  # Aggregate features

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)             # Flatten
        reduced = F.relu(self.fc_reduce(x))      # (batch, 256)
        tokens = reduced.view(batch_size, self.num_tokens, self.token_dim)  # (batch, 16, 16)
        attn_out = self.attention(tokens)        # (batch, 16, 16)
        aggregated = attn_out.mean(dim=1)        # (batch, 16)
        combined = torch.cat([reduced, aggregated.repeat(1, 16)], dim=1)  # (batch, 512)
        features = self.fc_out(combined)         # (batch, 64)
        return features

class EndToEndModel(nn.Module):
    def __init__(self, input_dim, scale_factor, num_classes):
        super(EndToEndModel, self).__init__()
        self.feature_extractor = AdvancedFeatureExtractor(input_dim, scale_factor)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

# ------------------------------
# Training on Multiple Input Datasets using ConcatDataset
# ------------------------------
def train_model_on_multiple_datasets(config, device):
    dataset_params = config["data"]
    input_dim = np.prod(dataset_params["image_shape"])
    num_classes = dataset_params["num_classes"]
    scale_factor = config["model"]["scale_factor"]

    # Simulate multiple datasets; in practice, replace these with your real datasets.
    dataset1 = SimulatedImageDataset(dataset_params["num_samples"],
                                     dataset_params["image_shape"],
                                     num_classes, dataset_id=1)
    dataset2 = SimulatedImageDataset(dataset_params["num_samples"],
                                     dataset_params["image_shape"],
                                     num_classes, dataset_id=2)
    dataset3 = SimulatedImageDataset(dataset_params["num_samples"],
                                     dataset_params["image_shape"],
                                     num_classes, dataset_id=3)
    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])
    dataloader = DataLoader(combined_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    model = EndToEndModel(input_dim, scale_factor, num_classes).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting training on multiple datasets...")
    for epoch in range(config["training"]["num_epochs"]):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(combined_dataset)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    torch.save(model.state_dict(), "multi_dataset_model.pth")
    logger.info("Training on multiple datasets completed and model saved.")
    return model

# ------------------------------
# Compression, Encryption, and HMAC Functions (unchanged)
# ------------------------------
def compress_features(features):
    features_bytes = json.dumps(features.tolist()).encode('utf-8')
    compressed = zlib.compress(features_bytes)
    logger.info(f"Compressed data to {len(compressed)} bytes.")
    return compressed

def encrypt_data_aes_gcm(compressed_data, key):
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(compressed_data)
    logger.info("Data encrypted with AES-GCM.")
    return cipher.nonce + tag + ciphertext

def compute_hmac(data, hmac_key):
    return hmac.new(hmac_key, data, hashlib.sha256).digest()

# ------------------------------
# RF Transmitter Module (Synchronous Version; unchanged)
# ------------------------------
class RFTransmitter:
    def __init__(self, modulation, band_speed):
        self.modulation = modulation
        self.band_speed = band_speed

    def update_channel_conditions(self, snr):
        if snr > 20:
            self.modulation = "16-QAM"
        elif snr > 10:
            self.modulation = "QPSK"
        else:
            self.modulation = "BPSK"
        logger.info(f"RF: Modulation updated to {self.modulation} (SNR: {snr} dB)")

    def add_error_correction(self, packet, redundancy_factor):
        corrected = packet * redundancy_factor
        logger.info(f"RF: Error correction applied with redundancy factor {redundancy_factor}.")
        return corrected

    def format_packet(self, payload, packet_id=1, version=1, hmac_key=None):
        timestamp = int(time.time())
        payload_length = len(payload)
        crc = zlib.crc32(payload) & 0xffffffff
        header = struct.pack(">HBIIL", packet_id, version, timestamp, payload_length, crc)
        if hmac_key is None:
            raise ValueError("HMAC key required for packet formatting.")
        header_hmac = compute_hmac(header, hmac_key)
        full_packet = header + header_hmac + payload
        logger.info("RF: Packet formatted with header and HMAC.")
        return full_packet

    def transmit(self, packet):
        transmission_time = len(packet) * 8 / self.band_speed
        logger.info(f"RF: Transmitting {len(packet)} bytes using {self.modulation} at {self.band_speed} bps (simulated {transmission_time:.4f} sec).")
        logger.info("RF: Transmission complete.")

# ------------------------------
# Inference Pipeline (Synchronous; unchanged)
# ------------------------------
def inference_pipeline(model, rf_transmitter, encryption_key, hmac_key, device):
    raw_data = load_data(CONFIG["data"]["image_shape"])
    preprocessed = preprocess_data(raw_data)
    tensor_data = torch.from_numpy(preprocessed).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor_data)
    features = logits.squeeze(0).cpu().numpy()
    logger.info("Inference complete: Features extracted.")
    compressed = compress_features(features)
    encrypted = encrypt_data_aes_gcm(compressed, encryption_key)
    packet = rf_transmitter.format_packet(encrypted, packet_id=100, version=1, hmac_key=hmac_key)
    packet_ec = rf_transmitter.add_error_correction(packet, CONFIG["rf"]["redundancy_factor"])
    rf_transmitter.transmit(packet_ec)
    logger.info("Inference pipeline executed successfully.")

# ------------------------------
# Main Canvas: Mode Selection (Train / Inference)
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local Onboard Processing Canvas for Multi-Input Datasets")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], required=True,
                        help="Mode: 'train' for training on multiple datasets, 'inference' for inference")
    args = parser.parse_args()

    # Generate secure keys if not provided
    if CONFIG["security"]["encryption_key"] is None:
        CONFIG["security"]["encryption_key"] = get_random_bytes(32)
    if CONFIG["security"]["hmac_key"] is None:
        CONFIG["security"]["hmac_key"] = get_random_bytes(32)
    encryption_key = CONFIG["security"]["encryption_key"]
    hmac_key = CONFIG["security"]["hmac_key"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rf_transmitter = RFTransmitter(CONFIG["rf"]["modulation"], CONFIG["rf"]["band_speed"])
    rf_transmitter.update_channel_conditions(snr=15)

    dataset_params = CONFIG["data"]
    input_dim = np.prod(dataset_params["image_shape"])
    num_classes = dataset_params["num_classes"]
    scale_factor = CONFIG["model"]["scale_factor"]

    if args.mode == "train":
        model = train_model_on_multiple_datasets(CONFIG, device)
    elif args.mode == "inference":
        model = EndToEndModel(input_dim, scale_factor, num_classes).to(device)
        if os.path.exists("multi_dataset_model.pth"):
            model.load_state_dict(torch.load("multi_dataset_model.pth", map_location=device))
            logger.info("Loaded multi-dataset model weights.")
        else:
            logger.warning("No saved model found; using randomly initialized weights.")
        inference_pipeline(model, rf_transmitter, encryption_key, hmac_key, device)
    else:
        logger.error("Invalid mode selected.")

if __name__ == "__main__":
    main()
