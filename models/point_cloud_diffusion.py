import torch
import numpy as np
import torch.nn as nn
from models.deepsets import DeepSetsAtt
import torch.nn.functional as F

activation = nn.LeakyReLU(0.01)


class PCD(nn.Module):  # Point Cloud Diffusion
    """Score based generative model"""

    def __init__(self, params, factor=1):
        super().__init__()

        self.factor = factor
        self.num_feat = params.N_cell_feat
        self.num_cluster = params.N_clust_feat
        self.num_cond = params.NUM_COND
        self.num_embed = params.EMBED
        self.num_steps = params.MAX_STEPS

        self.ema = 0.999

        # Diffusion TimeSteps
        self.timesteps = torch.arange(0, self.num_steps + 1,
                dtype=torch.float32) / self.num_steps + 8e-3

        # Diffusion Parameters, for the denoising and score learning
        self.Set_alpha_beta_posterior()

        projection_dim = 16
        # Random Fourier Features, to concat with input
        self.projection = self.GaussianFourierProjection()

        self.loss_tracker = nn.MSELoss()
        # keras.metrics.Mean(name="loss")

        # linear1_input_size = self.num_embed + self.num_cluster + self.num_cond
        graph_emb_size = self.num_embed + self.num_cluster + self.num_cond
        cluster_emb_size = self.num_embed + self.num_cond

        self.activation = nn.LeakyReLU(0.01)

        # DEFNE GRAPH Embedding
        self.graph_embedding1 = Embedding(projection_dim, self.num_embed)
        self.graph_linear1 = nn.Linear(graph_emb_size, self.num_embed)
        self.graph_activation1 = self.activation

        # DEFINE CLUSTER Embedding
        self.cluster_embedding1 = Embedding(projection_dim, self.num_embed)
        self.cluster_linear1 = nn.Linear(cluster_emb_size, self.num_embed)
        self.cluster_activation1 = self.activation

        self.shape = (-1,1,1)
        self.ds_attention_layer = DeepSetsAtt(
            num_feat=self.num_embed,
            time_embedding_dim=self.num_embed,
            num_heads=1,
            num_transformer=8,
            projection_dim=64)
            # time_embedding=graph_conditional,

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
        cluster_inputs = torch.cat([cluster_conditional, inputs_cond], -1)
        cluster_conditional = self.cluster_linear1(cluster_inputs)
        cluster_conditional = self.activation(cluster_conditional)

        # Define DeepSets Attention Layers
        inputs, outputs = self.ds_attention_layer(training_data,
                                                 graph_conditional,
                                                 inputs_mask)

        # inputs, oututs = self.ds_attention_layer(inputs, graph_conditional,
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

    def GaussianFourierProjection(self):
        half_dim = self.num_embed // 4
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freq = torch.exp(-emb * torch.arange(0, half_dim, dtype=torch.float32))
        return freq

    def get_weights_function(self):
        def weights_init(m):
            # classname = m.__class__.__name__
            nn.init.normal_(m.weight.data, 0.0)
        return weights_init

    def train_step(self, inputs):
        part, jet, cond, mask = inputs
        part = part * mask

        random_t = torch.rand((cond.size(0), 1))

        # Assume get_logsnr_alpha_sigma is defined elsewhere
        _, alpha, sigma = get_logsnr_alpha_sigma(random_t)

        alpha_reshape = alpha.view(self.shape)
        sigma_reshape = sigma.view(self.shape)

        # Part processing
        z = torch.randn(part.size(), dtype=torch.float32) * mask
        perturbed_x = alpha_reshape * part + z * sigma_reshape
        pred = self.model_part(perturbed_x * mask, random_t, jet, cond, mask)
        v = alpha_reshape * z - sigma_reshape * part
        losses = torch.square(pred - v) * mask
        loss_part = losses.view(losses.size(0), -1).mean()

        # Calculate gradients for part model
        self.optimizer.zero_grad()
        loss_part.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model_part.parameters(), 1)
        self.optimizer.step()

        # Jet processing
        z = torch.randn(jet.size(), dtype=torch.float32)
        perturbed_x = alpha * jet + z * sigma
        pred = self.model_jet(perturbed_x, random_t, cond)
        v = alpha * z - sigma * jet
        losses = torch.square(pred - v)
        loss_jet = losses.view(losses.size(0), -1).mean()

        # Calculate gradients for jet model
        self.optimizer.zero_grad()
        loss_jet.backward()
        torch.nn.utils.clip_grad_norm_(self.model_jet.parameters(), 1)
        self.optimizer.step()

        return {
            "loss": loss_jet + loss_part,
            "loss_part": loss_part.item(),
            "loss_jet": loss_jet.item(),
        }


    @torch.no_grad()  # Ensure this does not track gradients
    def val_step(self, inputs):
        part, cluster, cond, mask = inputs

        random_t = torch.randint(0, self.num_steps,
                                 (cond.size(0), 1),
                                 dtype=torch.int32)

        alpha = torch.gather(self.alphas_cumprod, 0, random_t)
        sigma = torch.sqrt(1 - self.alphas_cumprod[random_t])

        alpha_reshape = alpha.view(self.shape)
        sigma_reshape = sigma.view(self.shape)

        # Part
        z = torch.randn(part.size(), dtype=torch.float32)
        perturbed_x = alpha_reshape * part + z * sigma_reshape
        score = self.model_part(perturbed_x, random_t, cluster, cond, mask)
        v = alpha_reshape * z - sigma_reshape * part
        losses = torch.square(score - v) * mask
        loss_part = losses.view(losses.size(0), -1).mean()

        # Cluster
        z = torch.randn(cluster.size(), dtype=torch.float32)
        perturbed_x = alpha * cluster + z * sigma
        score = self.model_cluster(perturbed_x, random_t, cond)
        v = alpha * z - sigma * cluster
        losses = torch.square(score - v)
        loss_cluster = losses.view(losses.size(0), -1).mean()

        self.loss_tracker += (loss_cluster + loss_part)

        return {
            "loss": self.loss_tracker.item() / (self.loss_tracker.numel()),
            "loss_part": loss_part.item(),
            "loss_cluster": loss_cluster.item(),
        }
    # End PCD Class

def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma

def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    logsnr_max_tensor = torch.tensor(logsnr_max, dtype=torch.float32)
    logsnr_min_tensor = torch.tensor(logsnr_min, dtype=torch.float32)
    b = torch.atan(torch.exp(-0.5 * logsnr_max_tensor))
    a = torch.atan(torch.exp(-0.5 * logsnr_min_tensor)) - b
    return -2. * torch.log(torch.tan(a * t.to(torch.float32) + b))

def inv_logsnr_schedule_cosine(self, logsnr, logsnr_min=-20., logsnr_max=20.):
        b = torch.atan(torch.exp(-0.5 * logsnr_max))
        a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
        return torch.atan(torch.exp(-0.5 * logsnr.to(torch.float32)))/a - b/a



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
    def get_weights_function(self):
        def weights_init(m):
            # classname = m.__class__.__name__
            nn.init.normal_(m.weight.data, 0.0)
        return weights_init
