import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer

# -------------------------
# Title Encoder (DeBERTa-v3)
# -------------------------
class DebertaTitleEncoder(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", output_dim=256, freeze_encoder=True, dropout=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def masked_mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / counts

    def forward(self, titles):
        inputs = self.tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        token_embeddings = outputs.last_hidden_state
        pooled = self.masked_mean_pool(token_embeddings, inputs["attention_mask"])
        return self.projection(pooled)

# -------------------------
# Channel Metadata Encoder
# -------------------------
class ChannelMetadataEncoder(nn.Module):
    def __init__(self, num_niches, num_languages, embed_dim=32, dropout=0.2):
        super().__init__()
        self.niche_embed = nn.Embedding(num_niches, 8)
        self.language_embed = nn.Embedding(num_languages, 4)
        self.continuous_dim = 6  # log_subs, avg_views_30d, avg_ctr_30d, uploads_per_week, channel_age_days, is_verified
        self.mlp = nn.Sequential(
            nn.Linear(8 + 4 + self.continuous_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim)
        )

    def forward(self, niche_id, language_id, continuous_features):
        niche_emb = self.niche_embed(niche_id)
        lang_emb = self.language_embed(language_id)
        x = torch.cat([niche_emb, lang_emb, continuous_features], dim=-1)
        return self.mlp(x)

# -------------------------
# Video Metadata Encoder
# -------------------------
class VideoMetadataEncoder(nn.Module):
    def __init__(self, embed_dim=32, dropout=0.2):
        super().__init__()
        # Categorical embeddings
        self.hour_embed = nn.Embedding(24, 8)
        self.dow_embed = nn.Embedding(7, 4)
        self.mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim)
        )

    def forward(self, publish_hour, day_of_week):
        hour_emb = self.hour_embed(publish_hour)
        dow_emb = self.dow_embed(day_of_week)
        x = torch.cat([hour_emb, dow_emb], dim=-1)
        return self.mlp(x)

# -------------------------
# Hybrid Model
# -------------------------
class HybridEvaluator(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        num_classes: int,
        backbone_name: str = "resnet50",
        device="cpu"
    ):
        super().__init__()
        self.device = device

        # Image Encoder
        if backbone_name == "resnet50":
            self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            img_feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif backbone_name == "resnet18":
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            img_feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Title Encoder
        self.title_encoder = DebertaTitleEncoder()
        title_feat_dim = 256

        # Metadata Encoders
        self.video_encoder = VideoMetadataEncoder(embed_dim=32)

        # Numeric / continuous features (optional)
        self.numeric_net = nn.Sequential(
            nn.BatchNorm1d(num_numeric_features),
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion + Prediction
        fused_dim = img_feature_dim + title_feat_dim + 32 + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        images,
        titles,
        numeric_features,
        video_publish_hour,
        video_day_of_week,
    ):
        # Image
        img_feat = self.cnn(images)
        # Title
        title_feat = self.title_encoder(titles)
        # Metadata
        video_feat = self.video_encoder(video_publish_hour, video_day_of_week)
        # Numeric
        num_feat = self.numeric_net(numeric_features)
        # Fuse all
        fused = torch.cat([img_feat, title_feat, video_feat, num_feat], dim=1)
        return self.head(fused)

class HybridVideoDataset(Dataset):
    def __init__(
        self,
        df,
        thumbnail_tensors,
        numeric_features,
        title_col,
        hour_col,
        dow_col,
        label_col,
    ):
        """
        df: pandas DataFrame
        thumbnail_tensors: Tensor [N, 3, H, W]
        numeric_features: list of column names
        """
        assert len(df) == thumbnail_tensors.shape[0]

        self.df = df.reset_index(drop=True)
        self.thumbnails = thumbnail_tensors
        self.numeric_features = numeric_features

        self.title_col = title_col
        self.hour_col = hour_col
        self.dow_col = dow_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "image": self.thumbnails[idx],                          # Tensor [3,H,W]
            "title": str(self.df.loc[idx, self.title_col]),         # string
            "numeric": torch.tensor(
                self.df.loc[idx, self.numeric_features].to_numpy(dtype=np.float32),
                dtype=torch.float32
            ),                                                       # Tensor [N]
            "hour": torch.tensor(
                self.df.loc[idx, self.hour_col],
                dtype=torch.long
            ),
            "dow": torch.tensor(
                self.df.loc[idx, self.dow_col],
                dtype=torch.long
            ),
            "label": torch.tensor(
                self.df.loc[idx, self.label_col],
                dtype=torch.long
            )
        }

def hybrid_collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),     # [B,3,H,W]
        "titles": [b["title"] for b in batch],                  # list[str]
        "numeric": torch.stack([b["numeric"] for b in batch]),  # [B,N]
        "hour": torch.stack([b["hour"] for b in batch]),        # [B]
        "dow": torch.stack([b["dow"] for b in batch]),          # [B]
        "labels": torch.stack([b["label"] for b in batch])      # [B]
    }
