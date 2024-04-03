import torch
import numpy as np
import yaml
import torch.nn as nn
from deepsets import DeepSetsAtt
import torch.nn.functional as F

activation = nn.LeakyReLU(0.01)

class PCD(nn.Module):  # Point Cloud Diffusion
    """Score based generative model"""

    def __init__(self, name='SGM', npart=30, config_file=None, factor=1):
        super(PCD, self).__init__()

        if config_file is None:
            raise ValueError("\nNeed to specify config file!!\n")

        config = yaml.safe_load(open(config_file))
        self.config = config

        self.factor = factor
        self.num_feat = self.config['NUM_FEAT']
        self.num_cluster = self.config['NUM_CLUS']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema = 0.999

        # Diffusion TimeSteps
        self.timesteps = torch.arange(0, self.num_steps + 1, 
                dtype=torch.float32) / self.num_steps + 8e-3

        # Diffusion Parameters, for the denoising and score learning
        self.Set_alpha_beta_posterior()

        self.projection = self.GaussianFourierProjection(scale=16)

        self.loss_tracker = nn.MSELoss()
        # keras.metrics.Mean(name="loss")

        # DEFINE THE INPUT LAYERS
        self.inputs_time = torch.tensor(1)
        self.inputs_cond = torch.tensor((self.num_cond))
        self.inputs_cluster = torch.tensor(self.num_cluster)
        self.inputs_mask = torch.tensor((1, 1)) # second eleme = feature size
        # ^TF version in (None,1). Torch can handle dynamic
        # input size, but the feature size (second dim)
        # must match. The mask has feature dim = 1

        # Activation function
        self.activation = nn.LeakyReLU(0.01)

        # DEFNE GRAPH MODEL
        self.graph_embedding1 = Embedding(self.num_embed)
        self.graph_linear1 = nn.Linear(self.num_embed, self.num_embed)
        self.graph_activation1 = self.activation

        # DEFINE CLUSTER MODEL
        self.cluster_embedding1 = Embedding(self.num_embed)
        self.cluster_linear1 = nn.Linear(self.num_embed, self.num_embed)
        self.cluster_activation1 = self.activation

        # the first dim will be n_cluster or n_part

    def forward(self, input_data):

        # Graph Forward Pass
        graph_conditional = self.graph_embedding1(self.inputs_time, self.projection)
        graph_inputs = torch.cat([graph_conditional,self.inputs_cluster,self.inputs_cond],-1)
        graph_conditional = nn.Linear(graph_inputs, self.num_embed)
        graph_conditional = self.activation(graph_conditional)

        # Cluster Forward Pass
        cluster_conditional = self.cluster_embedding1(self.inputs_time, self.projection)
        cluster_inputs = torch.cat([cluster_conditional, self.inputs_cond], -1)
        cluster_conditional = nn.Linear(cluster_inputs, self.num_embed)
        cluster_conditional = self.activation(cluster_conditional)

        inputs, outputs = DeepSetsAtt(
            num_feat=self.num_embed,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer=8,
            projection_dim=64,
            mask=self.inputs_mask,
        )

        # self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_cluster,inputs_cond,inputs_mask],outputs=outputs)

        return outputs

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


class Embedding(nn.Module):
    def __init__(self, num_embed):
        super(Embedding, self).__init__()

        self.num_embed = num_embed
        self.dense1 = nn.Linear(2 * self.num_embed, 2 * self.num_embed)
        self.dense2 = nn.Linear(2 * self.num_embed, self.num_embed)


    def forward(self, inputs, projection):

        angle = inputs * projection
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)

        embedding = self.dense1(embedding)
        embedding = F.leaky_relu(embedding, 0.01)
        embedding = self.dense2(embedding)
        return embedding


        # angle shape = (None, 16)
        # Num embed =  64
        # SIZE after concat sin, cos FUNCTION =  (None, 32)
        # SIZE of Dense Layer =  (None, 128)
        # num_embed 64
        # SIZE dense(num_embed)*embedding=  (None, 64)

    # def count_parameters(model):
    #     total_params = sum(p.numel() for p in model.parameters())
    #     return total_params

    # num_params = count_parameters(self.model_part)
    # print("Number of parameters: {:,}".format(num_params))
    # Number of parameters: 314,372
