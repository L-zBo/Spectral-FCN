import os
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.se(self.avg_pool(x))
        max_out = self.se(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ConvWide(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        return x


class ConvMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        out_channels = out_channels // 4
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, 4, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 3, 4, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, 5, 4, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, 7, 4, padding=3)
        self.norm = nn.BatchNorm1d(out_channels * 3)
        self.relu = nn.ReLU()
        self.ca = ChannelAttention(out_channels * 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.norm(x)
        x = self.relu(x)
        x = self.ca(x) * x
        x = torch.cat([x1, x], dim=1)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvWide(1, 64, kernel_size=8, stride=4),
            ConvMultiScale(64, 128),
            ConvMultiScale(128, 128),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.gap(x)
        return x.squeeze(-1)

    def load_weights(self, path, map_location='cpu'):
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=map_location)
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded FeatureEncoder weights from {path}")
            except Exception as e:
                print(f"Failed to load FeatureEncoder weights: {e}")
        else:
            print(f"FeatureEncoder weights not found at {path}, using random initialization.")


class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class FaultClassificationNetwork(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

    def save_weights(self, weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{weights_dir}/encoder.pth")
        torch.save(self.classifier.state_dict(), f"{weights_dir}/classifier.pth")

    def load_weights(self, weights_dir):
        self.encoder.load_state_dict(torch.load(f"{weights_dir}/encoder.pth", map_location="cpu"))
        self.classifier.load_state_dict(torch.load(f"{weights_dir}/classifier.pth", map_location="cpu"))


class FaultMultiHeadNetwork(nn.Module):
    def __init__(self, num_classes_pp=3, num_classes_pe=3, input_dim=128):
        super().__init__()
        self.encoder = FeatureEncoder()
        self.classifier_pp = Classifier(input_dim=input_dim, num_classes=num_classes_pp)
        self.classifier_pe = Classifier(input_dim=input_dim, num_classes=num_classes_pe)

    def forward(self, x):
        feat = self.encoder(x)
        pp_logits = self.classifier_pp(feat)
        pe_logits = self.classifier_pe(feat)
        return pp_logits, pe_logits

    def save_weights_multi(self, weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{weights_dir}/encoder.pth")
        torch.save(self.classifier_pp.state_dict(), f"{weights_dir}/classifier_pp.pth")
        torch.save(self.classifier_pe.state_dict(), f"{weights_dir}/classifier_pe.pth")

    def load_weights_multi(self, weights_dir, map_location="cpu"):
        self.encoder.load_state_dict(torch.load(f"{weights_dir}/encoder.pth", map_location=map_location))
        self.classifier_pp.load_state_dict(torch.load(f"{weights_dir}/classifier_pp.pth", map_location=map_location))
        self.classifier_pe.load_state_dict(torch.load(f"{weights_dir}/classifier_pe.pth", map_location=map_location))


if __name__ == "__main__":
    model = FaultClassificationNetwork(num_classes=4)
    test_input = torch.randn(32, 1, 273)
    output = model(test_input)
    print("Output shape:", output.shape)

    multi_model = FaultMultiHeadNetwork(num_classes_pp=3, num_classes_pe=3)
    pp_logits, pe_logits = multi_model(test_input)
    print("Multi-head PP logits shape:", pp_logits.shape)
    print("Multi-head PE logits shape:", pe_logits.shape)
