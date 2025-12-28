import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


class HybridCNNTabular(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        num_classes: int,
        backbone_name: str = "resnet50",
    ):
        super().__init__()

        # -------------------------
        # Image Encoder (CNN)
        # -------------------------
        if backbone_name == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            img_feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()  # remove classifier
        elif backbone_name == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            img_feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # -------------------------
        # Numeric Feature Encoder
        # -------------------------
        self.numeric_net = nn.Sequential(
            nn.BatchNorm1d(num_numeric_features),
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Fusion + Prediction Head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(img_feature_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, numeric_features: torch.Tensor):
        """
        images: Tensor [B, 3, H, W]
        numeric_features: Tensor [B, num_numeric_features]
        """
        img_feat = self.cnn(images)                 # [B, img_feature_dim]
        num_feat = self.numeric_net(numeric_features)  # [B, 64]

        fused = torch.cat([img_feat, num_feat], dim=1)
        output = self.head(fused)

        return output

class DummyHybridDataset(Dataset):
    def __init__(self, images, numbers, labels):
        self.images = images
        self.numbers = numbers
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.numbers[idx], self.labels[idx]


device = "cuda" if torch.cuda.is_available() else "cpu"

model = HybridCNNTabular(
    num_numeric_features=10,
    num_classes=5,
    backbone_name="resnet50",
).to(device)

# Dummy inputs
images = torch.randn(8, 3, 224, 224)
numbers = torch.randn(8, 10)
labels = torch.randint(0, 5, (8,))

outputs = model(images, numbers)
print(outputs.shape)  # torch.Size([8, 5])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

dataset = DummyHybridDataset(images, numbers, labels)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model.train()

for images, numbers, labels in dataloader:
    images = images.to(device)
    numbers = numbers.to(device)
    labels = labels.to(device)

    outputs = model(images, numbers)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
