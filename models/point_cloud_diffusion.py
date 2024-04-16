import torch
import numpy as np
import yaml
import torch.nn as nn
from deepsets import DeepSetsAtt
import torch.nn.functional as F
import keras.backend as K

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

        projection_dim = 16
        self.projection = self.GaussianFourierProjection(scale=projection_dim)

        self.loss_tracker = nn.MSELoss()
        # keras.metrics.Mean(name="loss")

        # DEFINE THE INPUT LAYERS
        # self.inputs_time = torch.tensor(1)
        # self.inputs_cond = torch.tensor((self.num_cond))
        # self.inputs_cluster = torch.tensor(self.num_cluster)
        # self.inputs_cluster = torch.tensor((2,2))
        # self.inputs_mask = torch.tensor((1, 1))
        # ^^^^^^^
        # In torch, the only the input and output dims need  to be
        # Specified. This is num_feat, num_clust, and cond (mask).
        # So we don't need these lines, but do need the size, 
        # obtained from the config file, "input_size" below

        # linear1_input_size = self.num_embed + self.num_cluster + self.num_cond
        graph_LinearInput_size = self.num_embed + self.num_cluster + self.num_cond
        cluster_LinearInput_size = self.num_embed + self.num_cond

        self.activation = nn.LeakyReLU(0.01)

        # DEFNE GRAPH MODEL
        self.graph_embedding1 = Embedding(projection_dim, self.num_embed)
        print("\n\ngraph_embedding1 = ",self.graph_embedding1)
        self.graph_linear1 = nn.Linear(graph_LinearInput_size, self.num_embed)  
        self.graph_activation1 = self.activation

        # DEFINE CLUSTER MODEL
        self.cluster_embedding1 = Embedding(projection_dim, self.num_embed)
        self.cluster_linear1 = nn.Linear(cluster_LinearInput_size, self.num_embed)
        self.cluster_activation1 = self.activation

        # self.ds_attention_layer = DeepSetsAtt(
        #     num_feat=self.num_embed,
        #     time_embedding=graph_conditional,
        #     num_heads=1,
        #     num_transformer=8,
        #     projection_dim=64)

    def forward(self,
                training_data,
                inputs_time,
                inputs_cluster,
                inputs_cond,
                inputs_mask):

        # Prababyl need to add more inputs to the forward function here.
        # IN Torch, it doesn't make sense to initialize the input like in TF to None
        # Simply set the correct size (from the config) in __init__,
        # And then pass the input_data, and input_times + cluster stuff

        # Graph Forward Pass
        graph_conditional = self.graph_embedding1(inputs_time, self.projection)
        graph_inputs = torch.cat([graph_conditional,inputs_cluster,inputs_cond],-1)
        graph_conditional = self.graph_linear1(graph_inputs)
        graph_conditional = self.activation(graph_conditional)


        # Cluster Forward Pass
        cluster_conditional = self.cluster_embedding1(inputs_time, self.projection)
        cluster_inputs = torch.cat([cluster_conditional, inputs_cond],-1)
        cluster_conditional = self.cluster_linear1(cluster_inputs)
        cluster_conditional = self.activation(cluster_conditional)

        outputs = graph_conditional*cluster_conditional

        # Define DeepSets Attention Layers
        # inputs, oututs = self.ds_attention_layer(inputs, 
        #                                         inputs_mask)


        # print(f"inputs = {inputs}")  # TF: (None, None, 4)
        # print(f"outputs1 = {outputs}") # TF: (None, None, 4)



        # self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_cluster,inputs_cond,inputs_mask],outputs=outputs)

        return outputs

    def Set_alpha_beta_posterior(self):
        # Some math behind diffusion

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
    def __init__(self, projection_dim, num_embed):
        super(Embedding, self).__init__()

        self.num_embed = num_embed
        self.dense1 = nn.Linear(2 * projection_dim, 2 * num_embed)
        self.dense2 = nn.Linear(2 * num_embed,          num_embed)


    def forward(self, inputs, projection):

        angle = inputs * projection
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)

        embedding = self.dense1(embedding)
        embedding = F.leaky_relu(embedding, 0.01)
        embedding = self.dense2(embedding)
        print(np.shape(embedding))

        return embedding


    # def count_parameters(model):
    #     total_params = sum(p.numel() for p in model.parameters())
    #     return total_params

    # num_params = count_parameters(self.model_part)
    # print("Number of parameters: {:,}".format(num_params))
    # Number of parameters: 314,372
