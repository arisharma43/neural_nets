from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 256,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): hidden dimension for the MLP layers
        """
        # This code was written by GitHub Copilot
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input dimension: 2 tracks * n_track points * 2 coordinates each
        input_dim = 2 * n_track * 2
        output_dim = n_waypoints * 2
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # This code was written by GitHub Copilot
        batch_size = track_left.shape[0]
        
        # Concatenate left and right tracks and flatten
        # Shape: (b, n_track, 2) + (b, n_track, 2) -> (b, 2*n_track*2)
        track_features = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        track_features = track_features.view(batch_size, -1)  # (b, 2*n_track*2)
        
        # Pass through MLP
        waypoints_flat = self.mlp(track_features)  # (b, n_waypoints*2)
        
        # Reshape to waypoints format
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        # This code was written by GitHub Copilot
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Query embeddings for waypoints (learnable)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Project 2D track points to d_model dimension
        self.track_projection = nn.Linear(2, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection to 2D waypoints
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # This code was written by GitHub Copilot
        batch_size = track_left.shape[0]
        
        # Concatenate left and right tracks
        track_points = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        
        # Project track points to d_model dimension
        track_features = self.track_projection(track_points)  # (b, 2*n_track, d_model)
        
        # Get query embeddings for waypoints
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        query_embeds = self.query_embed(query_indices)  # (n_waypoints, d_model)
        query_embeds = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)
        
        # Apply transformer decoder (cross-attention)
        # queries attend to track features
        waypoint_features = self.transformer_decoder(
            tgt=query_embeds,
            memory=track_features
        )  # (b, n_waypoints, d_model)
        
        # Project to 2D waypoints
        waypoints = self.output_projection(waypoint_features)  # (b, n_waypoints, 2)
        
        return waypoints


class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer Planner.
        You can use this directly in your ViTPlanner implementation.

        Args:
            h: height of input image
            w: width of input image
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        num_patches = (H // p) * (W // p)
        x = x.reshape(B, num_patches, C * p * p)

        # Linear projection
        return self.projection(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You can use the one you implemented in Homework 3.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        # This code was written by GitHub Copilot
        super().__init__()
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP layers
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        # This code was written by GitHub Copilot
        # First block: Layer norm -> Self-attention -> Residual
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(norm_x, norm_x, norm_x)
        x = x + attn_output
        
        # Second block: Layer norm -> MLP -> Residual
        norm_x = self.norm2(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        
        return x


class ViTPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        patch_size: int = 8,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        """
        Vision Transformer (ViT) based planner that predicts waypoints from images.

        Args:
            n_waypoints (int): number of waypoints to predict

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding and TransformerBlock classes.
        The input images are of size (96, 128).

        Hint: A typical ViT architecture consists of:
        1. Patch embedding layer to convert image into sequence of patches
        2. Positional embeddings (learnable parameters) added to patch embeddings
        3. Multiple transformer encoder blocks
        4. Final normalization layer
        5. Output projection to predict waypoints

        Hint: For this task, you can either:
        - Use a classification token ([CLS]) approach like in standard ViT as global image representation
        - Use learned query embeddings (similar to TransformerPlanner)
        - Average pool over all patch features
        """
        # This code was written by GitHub Copilot
        super().__init__()

        self.n_waypoints = n_waypoints
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(
            h=96, w=128, patch_size=patch_size, in_channels=3, embed_dim=embed_dim
        )
        
        # Positional embeddings for patches + CLS token
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Waypoint prediction heads
        self.waypoint_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, n_waypoints * 2)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, 96, 128) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)

        Hint: The typical forward pass consists of:
        1. Normalize input image
        2. Convert image to patch embeddings
        3. Add positional embeddings
        4. Pass through transformer blocks
        5. Extract features for prediction (e.g., [CLS] token or average pooling)
        6. Project to waypoint coordinates
        """
        # This code was written by GitHub Copilot
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        batch_size = x.shape[0]
        
        # Convert image to patch embeddings
        patch_embeds = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, patch_embeds], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Extract CLS token for global representation
        cls_features = x[:, 0]  # (B, embed_dim)
        
        # Predict waypoints
        waypoints_flat = self.waypoint_head(cls_features)  # (B, n_waypoints * 2)
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)  # (B, n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
