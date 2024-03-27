import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence


class DeepSetsAtt(nn.Module):
    def __init__(self, num_feat, time_embedding, num_heads=4, num_transformer=4, projection_dim=32):
        super(DeepSetsAtt, self).__init__()

        self.num_feat = num_feat
        self.time_embedding = time_embedding
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.projection_dim = projection_dim

        # Time Embedding Layers
        self.time_dense1 = nn.Linear(time_embedding.shape[-1], 2*projection_dim)
        self.time_dense2 = nn.Linear(2*projection_dim, projection_dim)

        # DeepSets Layers
        self.dense_proj = nn.Linear(num_feat + projection_dim, projection_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads)
        self.dense_output = nn.Linear(projection_dim, num_feat)

    def forward(self, inputs, mask=None):
        # Handling time embedding
        time = F.leaky_relu(self.time_dense1(self.time_embedding))
        time = self.time_dense2(time)
        time = time.unsqueeze(1).repeat(1, inputs.size(1), 1)

        # Concatenating time to inputs
        inputs_time = torch.cat((inputs, time), dim=-1)

        # Processing with DeepSets architecture
        tdd = F.leaky_relu(self.dense_proj(inputs_time))
        encoded_patches = tdd

        if mask is not None:
            mask_matrix = mask @ mask.transpose(-2, -1)
        else:
            mask_matrix = None

        for _ in range(self.num_transformer):

            encoded_patches = encoded_patches.permute(1, 0, 2)
            attention_output, _ = self.multihead_attn(encoded_patches, encoded_patches,
                                                      encoded_patches, attn_mask=mask_matrix)
            attention_output = attention_output.permute(1, 0, 2)
            x2 = attention_output + encoded_patches

            x3 = F.gelu(self.dense_proj(x2))
            x3 = F.gelu(self.dense_proj(x3))
            encoded_patches = x3 + x2

        representation = F.leaky_relu(self.dense_proj(tdd + encoded_patches))
        outputs = self.dense_output(representation)

        return inputs, outputs
