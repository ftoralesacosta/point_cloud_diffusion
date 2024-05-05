import os
import json
import yaml
import h5py as h5
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, DistributedSampler
from icecream import ic
# from plotting import *  #keep for plotting part of loader

np.random.seed(0)  # fix the seed to keep track of validation split

# names = ['g','q','t','w','z']
names = ['P', 'Theta']

labels200 = {

    'G4_5x5_smeared.h5': 0,

    }

labels1000 = {

    'log10_Uniform_03-23.hdf5': 0,
}

nevts = -1

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def get_data_loader(params, world_rank, device=0):
    train_data = ColliderDataset(params, validation=False)
    val_data   = ColliderDataset(params, validation=True )

    train_sampler = DistributedSampler(train_data)
    val_sampler   = DistributedSampler( val_data )


    train_loader = DataLoader(train_data,
                              batch_size=params.local_batch_size,
                              num_workers=params.num_data_workers,
                              sampler=train_sampler,
                              worker_init_fn=worker_init,
                              persistent_workers=True,
                              pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_data,
                            batch_size=params.local_batch_size,
                            num_workers=params.num_data_workers,
                            sampler=val_sampler,
                            worker_init_fn=worker_init,
                            pin_memory=torch.cuda.is_available())

    return train_loader, val_loader

class ColliderDataset(Dataset):

    def __init__(self, 
                 params,
                 validation=False,
                 rank=0,
                 size=1,
                 ncluster_var=2,
                 num_condition=2,  #genP,genTheta
                 batch_size=64,
                 make_tf_data=True,
                 split=0.8):

        self.n_cluster_feat = params.n_clust_feat
        self.n_cond = 2
        self.inmem = True  # loads all data into memory
        self.data_path = params.data_path

        # Files names based on n_particels param
        if params.npart == 200:
            params.files = ['G4_5x5_smeared.h5']
        elif params.npart == 1000:
            params.files = ['log10_Uniform_03-23.hdf5']

        if not validation:
            self.event_fraction = params.train_fraction
        else:
            self.event_fraction = params.val_fraction

        #FIXME: add test case!

        print(params.files[0])
        self.file_name = params.files[0]
        # FIXME: for multiple h5 files, save the filenames, 

        n_total = params.N_samples
        if params.N_samples == -1:
            for file in params.files:
                with h5.File(os.path.join(params.data_path,
                                      self.file_name),'r') as h5file:

                    n_total += h5file['cluster'][:].shape[0]
                    params.update({"N_samples": n_total})

            #Need the size of the resulting partition for train/val
            #Need the total number of events in the hdf5 file for above
            #need the train/val fractions, for above, and slices

        self.n_val = int(params.N_samples*params.val_fraction)
        self.n_train = int(params.N_samples*params.train_fraction)# broadway express
        self.n_test = int(params.N_samples*params.train_fraction)

        if not validation:
            self.event_slice = np.s_[:self.n_train]

        else:
            self.event_slice = np.s_[self.n_train:self.n_train+self.n_val]

        #test
        # self.event_slice = \
        #np.s_[self.n_train+self.n_val:self.n_train+self.n_val+self.n_test]



        if self.inmem:
            ''' load all data into memory'''

            self.cells = []
            self.clusters = []
            self.conditions = []

            for file in params.files:
                with h5.File(os.path.join(params.data_path, file),"r") as h5f:

                    #FIXME: Below logic only works with single file
                    cell=h5f['hcal_cells'][self.event_slice].astype(np.float32)
                    cluster=h5f['cluster'][self.event_slice].astype(np.float32)

                    self.cells.append(cell)
                    self.clusters.append(cluster)
                    print(np.shape(self.cells))

            self.clusters = np.concatenate(self.clusters)
            self.cells = np.concatenate(self.cells)
            print(np.shape(self.cells))

            # Take Log10 of GenP
            self.clusters = np.where(self.clusters[:,0] > 0.0, 
                                     np.log10(self.clusters[:,0]), 0.0)

            # Take Log10 of Cell E
            self.cells[:,:,0] = np.where(self.cells[:,:,0] > 0.0,
                                         np.log10(self.cells[:,:,0]), 0.0 )


            print(np.shape(self.clusters))
            if params.save_json:
                save_norm_dict(self.cells, self.clusters, nmax=-1)

            print(np.shape(self.cells))
            self.clusters = torch.from_numpy(self.clusters)
            self.cells = torch.from_numpy(self.cells)

            print("SHAPE AFTER TORCH CONVERSION")
            print(self.clusters.shape)

            self.norm_dict = LoadJson(f'preprocessing_{npart}.json')

            self.cells, self.clusters = shuffle(self.particles, self.cells, random_state=0)

            # Maybe we should add preprocessing BACK as a class method
            self.cells, self.clusters, self.conditionals = preprocessing(self.cells, self.clusters, 
                                                                         self.norm_dict, self.n_cond)

            # cell data is 0-padded
            self.mask = np.expand_dims(particles[:,:,-1],-1)
            self.cells = self.cells[:,:,:-1]*mask


            #Finally, convert to Torch Tensor
            self.cells = torch.from_numpy(self.cells)
            self.clusters = torch.from_numpy(self.clusters)
            self.conditionals = torch.from_numpy(self.conditionals)
            self.mask = torch.from_numpy(self.mask)

            # FIXME: need to return part,jet,cond,mask

            print("#"*40)
            print("CELLs in DATALOADER = ",cells)
            print("-"*40)
            print("Clusters in DATALOADER = ",clusters)
            print("-"*40)
            print("Conditionals in DATALOADER = ",conditions)
            print("#"*40)


        else:
            ''' buffers for reading in hdf5 chunks'''
            self.clus_buff = np.zeros((batch_size,
                                       self.cluster_features),
                                      dtype=np.float32)

            self.cond_buff = np.zeros((batch_size,
                                       self.num_condition),
                                      dtype=np.float32)

            self.cell_buff = np.zeros((batch_size, 
                                       self.npart, 
                                       self.num_cell_feat), 
                                      dtype=np.float32)

    def _open_file(self):
        self.file = h5py.File(self.file_name, 'r')

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, index):

        if not self.file and not self.inmem:
            self._open_file()

        if self.inmem:

            return (self.cells[index], self.clusters[index], self.conditionals[index], self.mask[index])

        else:
            self.file['clusters'].read_direct(self.cluster_buff)

            self.file['clusters'].read_direct(self.cluster_buff,
                                              np.s_[index],
                                              np.s_[index])
            #FIXME: Finish implementing from disk. Depends on chunk of HDF5, and N files...


def save_norm_dict(cells, clusters, nmax=200_000):
    '''calulates mean and stdev from nmax events'''
    print(f"Saving min and max clusters/cells to JSON using {nmax} events")

    npart = cells.shape[1]

    mask = cells[:,:,-1] == 1 #saves array of BOOLS instead of ints
    ic(np.shape(mask))
    ic(np.shape(cells))
    print(f" calc_norm_dict L221: Masked \
    {np.shape(cells[mask])[0]} / {len(mask)} cells")

    data_dict = {
        'max_cluster':np.max(clusters[:,:],0).tolist(),
        'min_cluster':np.min(clusters[:,:],0).tolist(),

        'max_cell':np.max(cells[mask][:,:-1],0).tolist(), #-1 avoids mask
        'min_cell':np.min(cells[mask][:,:-1],0).tolist(),
    }                

    print(f"\n\nsaving to json preprocessing_{npart}.json")

    SaveJson(f'preprocessing_{npart}.json', data_dict)


def preprocessing(cells, clusters, data_dict, num_condition):

    ''' transforms features to standard normal distributions '''
    ''' we often want acces to this function outside of the class'''

    npart = cells.shape[1]

    cells=cells.reshape(-1,cells.shape[-1]) #flattens D0 and D1

    conditionals, clusters = conditionals_from_cluster(clusters)

    # normalize the data, mimics StandardScalar, or z-scoring
    clusters[:,:] = np.ma.divide(
        (clusters[:,:] - data_dict['min_cluster']),
        (np.array(data_dict['max_cluster']) - data_dict['min_cluster'])
    ).filled(0)        

    cells[:,:-1] = np.ma.divide(
        cells[:,:-1]-data_dict['min_cell'],
        np.array(data_dict['max_cell']) - data_dict['min_cell']
    ).filled(0)

    # make gaus-like. 
    clusters = logit(clusters)
    cells[:,:-1] = logit(cells[:,:-1])

    print(f"\nL Shape of Cells in DataLoader = {np.shape(cells)}") 
    print(f"\nL Cells in DataLoader = \n{cells[0,15:20,:]}")

    return cells.astype(np.float32), clusters.astype(np.float32), conditionals.astype(np.float32)
    # END _preprocessing


def conditionals_from_cluster(clusters, num_condition):

    conditions = clusters[:num_condition]
    clusters = clusters[num_condition:]

    return conditions, clusters



def logit(x):                            
    alpha = 1e-6
    x = alpha + (1 - 2*alpha)*x
    return np.ma.log(x/(1-x)).filled(0)


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def revert_npart(npart,max_npart):

    #Revert the preprocessing to recover the cell multiplicity
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(max_npart))
    x = npart*data_dict['std_cluster'][-1] + data_dict['mean_cluster'][-1]
    x = revert_logit(x)
    x = x * (data_dict['max_cluster'][-1]-data_dict['min_cluster'][-1]) \
        + data_dict['min_cluster'][-1]
    return np.round(x).astype(np.int32)

def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

def ReversePrep(cells, clusters, npart):

    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(npart))
    num_part = cells.shape[1]    
    cells=cells.reshape(-1,cells.shape[-1])
    mask=np.expand_dims(cells[:,3]!=0,-1) #for 4D cell, this is Z

    print("mask (b) in reverseprep = ",cells[:5,:])
    print("mask (e) in reverseprep = ",cells[-5:,:])
    # print(f"\ncells shape = {np.shape(cells)}\n")

    def _revert(x,name='cluster'):    
        x = x*data_dict['std_{}'.format(name)] + data_dict['mean_{}'.format(name)]
        x = revert_logit(x)
        x = x * (np.array(data_dict['max_{}'.format(name)]) -
            data_dict['min_{}'.format(name)]) + data_dict['min_{}'.format(name)]
        return x

    cells = _revert(cells,'cell')

    cells = (cells*mask).reshape(clusters.shape[0],num_part,-1)

    clusters = _revert(clusters,'cluster')
    clusters[:,-1] = np.round(clusters[:,-1]) #num cells

    return cells,clusters


def Recenter(particles):
    ''' FIXME: adapt for layer-normalized cellE '''

    px = particles[:,:,2]*np.cos(particles[:,:,1])
    py = particles[:,:,2]*np.sin(particles[:,:,1])
    pz = particles[:,:,2]*np.sinh(particles[:,:,0])

    jet_px = np.sum(px,1)
    jet_py = np.sum(py,1)
    jet_pz = np.sum(pz,1)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_phi = np.ma.arctan2(jet_py,jet_px).filled(0)
    jet_eta = np.ma.arcsinh(np.ma.divide(jet_pz,jet_pt).filled(0))

    particles[:,:,0]-= np.expand_dims(jet_eta,1)
    particles[:,:,1]-= np.expand_dims(jet_phi,1)


    return particles


def SimpleLoader(data_path,
                 labels,
                 ncluster_var=2,
                 num_condition=2):
    ''' not implemented yet. 5/2/2024'''

    cells = []
    clusters = []
    cond = []

    for label in labels:
        #if 'w' in label or 'z' in label: continue #no evaluation for w and z
        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['cluster'][:].shape[0]
            # ntotal = int(nevts)
            cell = h5f['hcal_cells'][int(0.7*ntotal):].astype(np.float32)
            cluster = h5f['cluster'][int(0.7*ntotal):].astype(np.float32)
            cluster = np.concatenate([cluster,labels[label]*np.ones(shape=(cluster.shape[0],1),dtype=np.float32)],-1)

            cells.append(cell)
            clusters.append(cluster)

    cells = np.concatenate(cells)
    clusters = np.concatenate(clusters)

    #Split Conditioned Features and Cluster Training Features

    cond = clusters[:,:num_condition] # GenP, GenTheta 
    clusters = clusters[:,ncluster_var:] # ClusterSum, N_Hits

    cells,clusters = shuffle(cells,clusters, random_state=0)

    mask = np.expand_dims(cells[:nevts,:,-1],-1)

    return cells[:nevts,:,:-1]*mask,clusters[:nevts],cond[:nevts]




