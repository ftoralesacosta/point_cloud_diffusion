import torch
import numpy as np
import yaml
import torch.nn as nn
from deepsets import DeepSetsAtt


class PCD(nn.Module):  # Point Cloud Diffusion
    """Score based generative model"""

    def __init__(self, name='SGM', npart=30, config_file=None, factor=1):
        super(PCD, self).__init__()

        if config_file is None:
            raise ValueError("\nNeed to specify config file!!\n")

        config = yaml.safe_load(open(config_file))
        self.config = config

        self.activation = nn.LeakyReLU(0.01)
        self.factor = factor
        self.num_feat = self.config['NUM_FEAT']
        self.num_cluster = self.config['NUM_CLUS']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema = 0.999

        # Diffusion TimeSteps
        self.timesteps = torch.arange(0, self.num_steps + 1, dtype=torch.float32) / self.num_steps + 8e-3

        # Diffusion Parameters, for the denoising and score learning
        self.Set_alpha_beta_posterior()
        print(self.alphas_cumprod)

        # learnable parameters, or layer that learns data projection
        self.projection = self.GaussianFourierProjection(scale=16)

    def Set_alpha_beta_posterior(self):
        # The math behind diffusion
        # Cumulative alphas, s.t. you don't
        # have to calculate noise at each step

        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = torch.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = torch.clamp(betas, min=0, max=0.999)
        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=torch.float32),
                                         alphas_cumprod[:-1]), 0)

        # set
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod

        self.posterior_variance = self.betas * \
            (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.posterior_mean_coef1 = (self.betas * \
            torch.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))

        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * \
            torch.sqrt(alphas) / (1. - self.alphas_cumprod)

    def GaussianFourierProjection(self, scale=30):
        half_dim = self.num_embed // 4
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freq = torch.exp(-emb * torch.arange(0, half_dim, dtype=torch.float32))
        return freq

    def Embedding(self, inputs, projection):
        angle = inputs * projection
        embedding = torch.cat([torch.sin(angle), torch.cos(angle)], -1)
        embedding = nn.Linear(2*self.num_embed, self.num_embed)(embedding)
        embedding = self.activation(embedding)
        return embedding

    def forward(self, inputs_time, inputs_cond, inputs_cluster, inputs_mask):
        self.projection = self.GaussianFourierProjection(scale=16)
        self.loss_tracker = nn.MSELoss()  # Example loss function

        graph_conditional = self.Embedding(inputs_time, self.projection)
        cluster_conditional = self.Embedding(inputs_time, self.projection)

        graph_conditional = nn.Linear(self.num_embed, self.num_embed)(
            torch.cat([graph_conditional, inputs_cluster, inputs_cond], -1))
        graph_conditional = self.activation(graph_conditional)

        cluster_conditional = nn.Linear(self.num_embed, self.num_embed)(
            torch.cat([cluster_conditional, inputs_cond], -1))
        cluster_conditional = self.activation(cluster_conditional)

        outputs = DeepSetsAtt(
            num_feat=self.num_embed,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer=8,
            projection_dim=64,
            mask=inputs_mask,
        )

        return outputs



    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    # num_params = count_parameters(self.model_part) 
    # print("Number of parameters: {:,}".format(num_params)) # Number of parameters: 314,372
