import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetsAtt(nn.Module):
    def __init__(self,
                 num_feat,
                 time_embedding_dim,
                 num_heads=4,
                 num_transformer=4,
                 projection_dim=32):

        super(DeepSetsAtt, self).__init__()

        self.time_dense1 = nn.Linear(time_embedding_dim, 2 * projection_dim)
        self.time_activation1 = nn.LeakyReLU(0.01)
        self.time_dense2 = nn.Linear(2 * projection_dim, projection_dim)

        self.patch_dense = nn.Linear(num_feat + projection_dim, projection_dim)
        self.patch_activation = nn.LeakyReLU(0.01)
        self.encoded_patches_dense = nn.Linear(projection_dim, projection_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads) for _ in range(num_transformer)
        ])

        self.final_dense = nn.Linear(projection_dim, projection_dim)
        self.final_activation = nn.LeakyReLU(0.01)
        self.output_dense = nn.Linear(projection_dim, num_feat)

    def forward(self, inputs, time_embedding, mask=None):
        # Process time embedding
        print(inputs.size())
        time_info = self.time_dense1(time_embedding)
        print(inputs.size())
        time_info = self.time_activation1(time_info)
        print(inputs.size())
        time_info = self.time_dense2(time_info)
        print(inputs.size())
        time_info = time_info.unsqueeze(1).repeat(1, inputs.size(1), 1)
        print(inputs.size())

        # Combine inputs and time information
        print("\n\n forward pass in deepsets.py")
        print(inputs.size())
        print(time_info.size())
        combined_inputs = torch.cat([inputs, time_info], dim=-1)

        tdd = self.patch_dense(combined_inputs)
        tdd = self.patch_activation(tdd)
        encoded_patches = self.encoded_patches_dense(tdd)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, inputs.size(1), -1)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            encoded_patches = block(encoded_patches, mask)

        representation = self.final_dense(tdd + encoded_patches)
        representation = self.final_activation(representation)
        outputs = self.output_dense(representation)

        return inputs, outputs


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.dense1 = nn.Linear(dim, 2 * dim)
        self.dense2 = nn.Linear(2 * dim, dim)

    def forward(self, x, attn_mask=None):
        x1 = self.norm1(x)
        x1, _ = self.attn(x1, x1, x1, attn_mask=attn_mask)
        x2 = x + x1
        x3 = self.norm2(x2)
        x3 = F.gelu(self.dense1(x3))
        x3 = F.gelu(self.dense2(x3))
        x3 += x2
        return x3

# Example usage:
# model = DeepSetsAtt(num_feat=4, time_embedding_dim=64)
# outputs = model(inputs_tensor, time_embedding_tensor, mask_tensor)

