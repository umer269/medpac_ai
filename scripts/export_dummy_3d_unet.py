"""
scripts/export_dummy_3d_unet.py
───────────────────────────────
Defines a standard 3D U-Net architecture in PyTorch and exports it to ONNX.
Run this if you want to test the 'onnx' model type path in config.yaml.

Requirements:
    pip install torch onnx
Run:
    python scripts/export_dummy_3d_unet.py
"""

import os

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    """
    Standard lightweight 3D U-Net architecture.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=8):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._conv_block(in_channels, init_features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._conv_block(init_features, init_features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(init_features * 2, init_features * 4)
        
        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(init_features * 4, init_features * 2)
        
        self.up1 = nn.ConvTranspose3d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(init_features * 2, init_features)
        
        # Output prediction layer (Sigmoid activation is applied during thresholding)
        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        # Bottleneck
        bn = self.bottleneck(pool2)
        
        # Decoder with Skip Connections
        up2 = self.up2(bn)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        # Output Sigmoid probabilites
        return torch.sigmoid(self.final(dec1))


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    model_path = "models/3d_unet_ventricles.onnx"
    
    print("\nInitializing PyTorch 3D U-Net model...")
    model = UNet3D(in_channels=1, out_channels=1, init_features=8)
    model.eval()
    
    # Dummy input representing (batch_size, channels, depth/slices, height, width)
    # The pipeline resamples scans to a standard grid size, e.g., 90x128x128.
    # We will export using dynamic axes so the model accepts any input volume shape.
    dummy_input = torch.randn(1, 1, 32, 64, 64)
    
    print(f"Exporting model to ONNX format at '{model_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "depth", 3: "height", 4: "width"},
            "output": {2: "depth", 3: "height", 4: "width"}
        }
    )
    print("✅ Export complete. You can now use model_type: 'onnx' in config.yaml!\n")
