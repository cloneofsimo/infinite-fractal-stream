import torch
import torch.nn as nn
import math


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
    ):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        self.num_layers = depth
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        nn.init.kaiming_normal_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        for layer in self.transformer_encoder.layers:
            nn.init.kaiming_normal_(layer.linear1.weight)
            nn.init.kaiming_normal_(layer.linear2.weight)
            nn.init.kaiming_normal_(layer.self_attn.in_proj_weight)
            nn.init.constant_(layer.self_attn.in_proj_bias, 0)
            nn.init.normal_(
                layer.self_attn.out_proj.weight,
                mean=0.0,
                std=0.2 / math.sqrt(self.embed_dim * self.num_layers),
            )
            nn.init.constant_(layer.self_attn.out_proj.bias, 0)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        N = x.size(1)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : N + 1, :]
        x = self.pos_drop(x)

        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)

        cls_token_final = x[:, 0]
        cls_token_final = self.norm(cls_token_final)

        logits = self.head(cls_token_final)

        return logits


def create_vit_model(image_size, patch_size, num_classes, embed_dim, depth, num_heads):
    model = VisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
    )
    return model


def test_create_vit_model():
    image_size = 224
    patch_size = 16
    num_classes = 1000
    embed_dim = 768
    depth = 12
    num_heads = 12

    model = create_vit_model(
        image_size, patch_size, num_classes, embed_dim, depth, num_heads
    )

    print(model)

    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        output = model(input_tensor)

    expected_output_shape = (batch_size, num_classes)
    assert (
        output.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, but got {output.shape}"

    print(f"Test passed! Output shape: {output.shape}")


if __name__ == "__main__":

    test_create_vit_model()
