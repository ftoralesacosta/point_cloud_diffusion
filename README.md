
### Fast Point Cloud Diffusion for EIC Calorimeters
This repository is a PyTorch implementation of the Fast Point Cload Diffusion repo for the EIC: https://github.com/ftoralesacosta/GSGM_for_EIC_Calo/tree/main.

Credit to Vinicius Mikuni for the original fast point cloud diffusion repoitory for generating jets:

https://github.com/ViniciusMikuni/GSGM_for_EIC_Calo

## This repo aims to improve on the previous projects in two ways:

1. The paper based on the old repo does not normalize the energy in each layer of the calorimeter. So the second diffusion model responsible for generating the point clouds in each layer must learn the absolute scale of  the energy in each layer, in additon to the various distributions of  the hits. This repo will add that additional normalization.
2. Switching to PyTorch for (hopefully) better legibility and modularity. Ideally, this could be used as a basis for larger point cloud diffusion projects, such as whole EIC event generation.


If you're reading this sentence, then this repo is still very much work in progress, and likely doesn't train just yet.
