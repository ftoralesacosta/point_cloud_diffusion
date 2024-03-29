import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
import utils
from deepsets import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu

tf.random.set_seed(1235)

class GSGM(keras.Model):
    """Score based generative model"""

    def __init__(self,name='SGM',npart=30,config=None,factor=1):

        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("\nConfig File Not Specified\n")


        self.activation = layers.LeakyReLU(alpha=0.01)
        self.factor=factor
        # Input Data parameters
        self.num_feat = self.config['NUM_FEAT']
        self.num_cluster = self.config['NUM_CLUS']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema=0.999

        # Diffusion parameters
        self.timesteps = tf.range(start=0,limit=self.num_steps + 1, dtype=tf.float32) / self.num_steps + 8e-3 
        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        self.betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(alphas, 0)
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - self.alphas_cumprod)
        self.projection = self.GaussianFourierProjection(scale = 16)

        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        self.loss_tracker = keras.metrics.Mean(name="loss")

        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond)) # shape=(None, 2) 2 
        inputs_cluster = Input((self.num_cluster)) # shape=(None, 2) 2 

        print("*"*10,"GSGM.py: 59","*"*10)
        print('inputs_cluster', inputs_cluster)
        print('inputs_cond',inputs_cond)



        # Defining the Model
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects       

        graph_conditional = self.Embedding(inputs_time,self.projection) # shape=(None, 64) print('graph_conditional',graph_conditional)
        cluster_conditional = self.Embedding(inputs_time,self.projection) # shape=(None, 64) print('cluster_conditional',cluster_conditional)      
        
        print('graph_conditional 0',graph_conditional) # shape=(None, 64)
        print('cluster_conditional 0',cluster_conditional) # shape=(None, 64)
        
        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,inputs_cluster,inputs_cond],-1))
        graph_conditional=self.activation(graph_conditional)
        
        cluster_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [cluster_conditional,inputs_cond],-1))
        cluster_conditional=self.activation(cluster_conditional)

        # These outputs for graph_conditional and cluster_conditional are (None,64) as they pass through embedding sizedense layer.
        print('graph_conditional 1',graph_conditional) # shape=(None, 64)
        print('cluster_conditional 1',cluster_conditional) # shape=(None, None, 64)
        # These conditionals will now get attached to Inputs
        # x1 + 

        # This block is only for cells, where the input x is of dim (None,None,4) (x,y,z,E)
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer = 8, #nominal 8
            projection_dim = 64,
            mask = inputs_mask,
        )

        print(f"inputs = {inputs}") # inputs = KerasTensor(type_spec=TensorSpec(shape=(None, None, 4), dtype=tf.float32, name='input_5'), name='input_5', description="created by layer 'input_5'")
        print(f"outputs1 = {outputs}") # outputs1 = KerasTensor(type_spec=TensorSpec(shape=(None, None, 4), dtype=tf.float32, name=None), name='time_distributed_5/Reshape_1:0', description="created by layer 'time_distributed_5'")
        print('num_cluster', self.num_cluster) # num_cluster 2
        
        # (None, None, 4){x,y,z,E} , (None, 64){time embedding_attaches to each particle differently} , cluster = (None, 2){Number, E_cl} , inputs_cond = (None, 2) {Pgen, Theta}, (None,1){For masking E value cell < 0}
        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_cluster,inputs_cond,inputs_mask],outputs=outputs)
        # outputs = (None, None, 4)
