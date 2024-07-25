import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382).

    Reference:
        https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                                   device=x.device)
            random_tensor.floor_()
            return x * random_tensor
        return x


class RandomDrop(nn.Module):
    def __init__(self, drop_prob: float, num_skip: float):
        super(RandomDrop, self).__init__()
        self.drop_prob = drop_prob
        self.num_skip = num_skip

    def forward(self, x, training=False):

        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0], 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                                   device=x.device)
            random_tensor.floor_()
            x[:, :, self.num_skip:] = x[:, :, self.num_skip:] * random_tensor[:, None]
            return x

        return x


class SimpleHeadAttention(nn.Module):
    """Simple MHA where masks can be directly added to the inputs.
    Args:

        projection_dim (int): projection dimension for the query,
        key, and value of attention.

        num_heads (int): number of attention heads.

        dropout_rate (float): dropout rate to be used for dropout in
        the attention scores as well as the final projected outputs.
    """
    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):

        super(SimpleHeadAttention, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.scale = (projection_dim // num_heads) ** -0.5
        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, int_matrix=None, mask=None, training=False):
        B, N, C = x.shape

        # Project the inputs all at once.
        qkv = self.qkv(x)

        # Reshape the projected output so that they're segregated in
        # terms of query, key, and value projections.

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Transpose so that the `num_heads` becomes the leading dimensions.
        # Helps to better segregate the representation sub-spaces.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        scale = self.scale
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        # Obtain the raw attention scores.
        attn = torch.matmul(q, k.transpose(-2, -1))

        # Add the integer matrix if provided.
        if int_matrix is not None:
            attn += int_matrix

        # Apply the mask if provided.
        if mask is not None:
            mask = mask.to(dtype=attn.dtype)
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn += (1.0 - mask) * -1e9

        # Normalize the attention scores.
        attn = self.softmax(attn)

        # Apply dropout to the attention scores.
        attn = self.attn_drop(attn)

        # Final set of projections as done in the vanilla attention mechanism.
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn



class TalkingHeadAttention(nn.Module):
    """Talking-head attention from CaiT: https://arxiv.org/abs/2003.02436.
    Args:
        projection_dim (int): projection dimension for the
        query, key, and value of attention.

        num_heads (int): number of attention heads.

        dropout_rate (float): dropout rate to be used for dropout
        in the attention scores as well as the final projected outputs.
    """

    def __init__(self, projection_dim: int, num_heads: int, dropout_rate: float):
        super(TalkingHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate

        head_dim = self.projection_dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(projection_dim, projection_dim)
        self.proj_l = nn.Linear(self.num_heads, self.num_heads)
        self.proj_w = nn.Linear(self.num_heads, self.num_heads)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, int_matrix=None, mask=None):
        B, N, C = x.shape

        # Project the inputs all at once.
        qkv = self.qkv(x)

        # Reshape the projected output so that they're segregated in terms of
        # query, key, and value projections.
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

        # Transpose so that the `num_heads` becomes the leading dimensions.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        scale = self.scale
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]

        # Obtain the raw attention scores.
        attn = torch.matmul(q, k.transpose(-2, -1))
        if int_matrix is not None:
            attn += int_matrix

        # Linear projection of the similarities between
        # the query and key projections.
        attn = attn.permute(0, 2, 3, 1)
        attn = self.proj_l(attn)

        # Normalize the attention scores.
        attn = attn.permute(0, 3, 1, 2)

        if mask is not None:
            mask = mask.to(dtype=attn.dtype)
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn += (1.0 - mask) * -1e9

        attn = self.softmax(attn)

        # Linear projection on the softmaxed scores.
        attn = attn.permute(0, 2, 3, 1)
        attn = self.proj_w(attn)
        attn = attn.permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        # Final set of projections as done in the vanilla attention mechanism.
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LayerScale(nn.Module):
    def __init__(self, init_values, projection_dim):
        super(LayerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim),
                                  requires_grad=True)

    def forward(self, inputs, mask=None):
        if mask is not None:
            return inputs * self.gamma * mask
        else:
            return inputs * self.gamma


# TESTS ------------------------------------


def test_simple_head_attention():
    batch_size = 2
    seq_length = 4
    projection_dim = 8
    num_heads = 2
    dropout_rate = 0.1

    model = SimpleHeadAttention(projection_dim, num_heads, dropout_rate)
    x = torch.randn(batch_size, seq_length, projection_dim)
    output, attn = model(x, training=True)

    # Check the output shape
    assert output.shape == (batch_size, seq_length, projection_dim),\
        f"Output shape mismatch: {output.shape}"

    # Check the attention shape
    assert attn.shape == (batch_size, num_heads, seq_length, seq_length),\
        f"Attention shape mismatch: {attn.shape}"
    print("SimpleHeadAttention test passed!")


def test_talking_head_attention():
    batch_size = 2
    seq_length = 4
    projection_dim = 8
    num_heads = 2
    dropout_rate = 0.1

    model = TalkingHeadAttention(projection_dim, num_heads, dropout_rate)
    x = torch.randn(batch_size, seq_length, projection_dim)
    output, attn = model(x)

    # Check the output shape
    assert output.shape == (batch_size, seq_length, projection_dim),\
        f"Output shape mismatch: {output.shape}"

    # Check the attention shape
    assert attn.shape == (batch_size, num_heads, seq_length, seq_length),\
        f"Attention shape mismatch: {attn.shape}"
    print("TalkingHeadAttention test passed!")


def test_layer_scale():
    batch_size = 2
    seq_length = 4
    projection_dim = 8
    init_values = 0.1

    model = LayerScale(init_values, projection_dim)
    inputs = torch.randn(batch_size, seq_length, projection_dim)
    mask = torch.ones(batch_size, seq_length, projection_dim)
    output = model(inputs, mask)

    # Check the output shape
    assert output.shape == (batch_size, seq_length, projection_dim),\
        f"Output shape mismatch: {output.shape}"

    print("LayerScale test passed!")

# Run the tests
# test_simple_head_attention()
# test_talking_head_attention()
