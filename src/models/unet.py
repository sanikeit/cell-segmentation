import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder pathway
        in_size = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_size, feature))
            in_size = feature

        # Enhance bottleneck with stronger regularization
        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder pathway
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Add dropout to decoder path
        self.decoder_dropout = nn.ModuleList([
            nn.Dropout2d(0.3) for _ in range(len(features))
        ])

        # Final convolution with extra regularization
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(features[0] // 2, out_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()  # Initialize sigmoid activation
        
        # Add deep supervision layers
        self.deep_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(f, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=2**(i+1))
            ) for i, f in enumerate(features)
        ])

    def forward(self, x):
        skip_connections = []
        deep_features = []
        input_size = x.shape[-2:]

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            deep_features.append(x)  # Store features before pooling
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder with dropout
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
            x = self.decoder_dropout[idx//2](x)

        # Main output
        main_output = self.final_conv(x)
        
        if self.training:
            # Deep supervision outputs
            aux_outputs = []
            for feat, deep_conv in zip(deep_features, self.deep_outputs):
                aux_out = deep_conv(feat)
                if aux_out.shape[-2:] != input_size:
                    aux_out = F.interpolate(aux_out, size=input_size, mode='bilinear', align_corners=True)
                aux_outputs.append(aux_out)
            
            return self.sigmoid(main_output), [self.sigmoid(aux) for aux in aux_outputs]
        
        return self.sigmoid(main_output)

def create_unet(config):
    """Factory function to create UNet with different configurations"""
    return UNet(
        in_channels=config.get('in_channels', 3),
        out_channels=config.get('out_channels', 1),
        features=config.get('features', [64, 128, 256, 512])
    )