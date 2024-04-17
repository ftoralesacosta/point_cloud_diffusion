import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import torch
from plotting import *
from torch.utils.data import Dataset, TensorDataset

# import tensorflow as tf
#from keras.utils.np_utils import to_categorical

# import energyflow as ef

np.random.seed(0)  # fix the seed to keep track of validation split

# names = ['g','q','t','w','z']
names = ['P', 'Theta']

labels200 = {

    # 'G4_smeared.h5':0,
    # 'log10_Uniform_03-23.hdf5':0,
    # 'G4_5x5_point_cloud.h5':0,
    # 'G4test_5x5_point_cloud.h5':0,
    'G4_5x5_smeared.h5':0,

    }

labels1000 = {
    'log10_Uniform_03-23.hdf5':0,
}

nevts = -1
# nevts = 500_000
# nevts = 100_000
# nevts = 8000

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
    x = x * (data_dict['max_cluster'][-1]-data_dict['min_cluster'][-1]) + data_dict['min_cluster'][-1]
    #x = np.exp(x)
    return np.round(x).astype(np.int32)

def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

def ReversePrep(cells,clusters,npart):

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
        x = x * (np.array(data_dict['max_{}'.format(name)]) -data_dict['min_{}'.format(name)]) + data_dict['min_{}'.format(name)]
        return x
        
    cells = _revert(cells,'cell')

    cells = (cells*mask).reshape(clusters.shape[0],num_part,-1)

    clusters = _revert(clusters,'cluster')
    clusters[:,-1] = np.round(clusters[:,-1]) #num cells

    return cells,clusters

def SimpleLoader(data_path,
                 labels,
                 ncluster_var = 2,
                 num_condition = 2):

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

import torch
from torch.utils.data import Dataset


class ZipDataSets(Dataset):
    ''' combines part, cluster, cond, and mask torch datasets'''

    def __init__(self, dataset1, dataset2, dataset3, dataset4):
        # Ensure all datasets have the same length
        assert len(dataset1) == len(dataset2) == len(dataset3) == len(dataset4)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return (self.dataset1[idx], self.dataset2[idx], self.dataset3[idx], self.dataset4[idx])

# Assuming part_dataset, cluster_dataset, cond_dataset, mask_dataset are PyTorch datasets
pytorch_zip = CombinedDataset(part_dataset, cluster_dataset, cond_dataset, mask_dataset)



def DataLoader(data_path,labels,
               npart,
               rank=0,size=1,
               ncluster_var=2,
               num_condition=2,#genP,genTheta
               batch_size=64,make_tf_data=True):
    cells = []
    clusters = []

    def _preprocessing(cells,clusters,save_json=False):
        num_part = cells.shape[1]

        cells=cells.reshape(-1,cells.shape[-1]) #flattens D0 and D1

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        #Transformations

        if save_json:
            mask = cells[:,-1] == 1 #saves array of BOOLS instead of ints
            print(f"L 357: Masked {np.shape(cells[mask])[0]} / {len(mask)} cells")
            #print(f"L 357: Masked {np.shape(cells[mask])[0]} / {len(mask)} cells") 

            data_dict = {
                'max_cluster':np.max(clusters[:,:],0).tolist(),
                'min_cluster':np.min(clusters[:,:],0).tolist(),

                # With Mask
                'max_cell':np.max(cells[mask][:,:-1],0).tolist(), #-1 avoids mask
                'min_cell':np.min(cells[mask][:,:-1],0).tolist(),

                # No Mask
                # 'max_cell':np.max(cells[:,:-1],0).tolist(),
                # 'min_cell':np.min(cells[:,:-1],0).tolist(),

            }                
            
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)

        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(npart))


        #normalize
        clusters[:,:] = np.ma.divide(clusters[:,:]-data_dict['min_cluster'],np.array(data_dict['max_cluster'])- data_dict['min_cluster']).filled(0)        
        cells[:,:-1]= np.ma.divide(cells[:,:-1]-data_dict['min_cell'],np.array(data_dict['max_cell'])- data_dict['min_cell']).filled(0)

        # make gaus-like. 
        clusters = _logit(clusters)
        cells[:,:-1] = _logit(cells[:,:-1])

        if save_json:
            mask = cells[:,-1]
            mean_cell = np.average(cells[:,:-1],axis=0,weights=mask)
            data_dict['mean_cluster']=np.mean(clusters,0).tolist()
            data_dict['std_cluster']=np.std(clusters,0).tolist()
            data_dict['mean_cell']=mean_cell.tolist()
            data_dict['std_cell']=np.sqrt(np.average((cells[:,:-1] - mean_cell)**2,axis=0,weights=mask)).tolist()                        

            SaveJson('preprocessing_{}.json'.format(npart),data_dict)


        clusters = np.ma.divide(clusters-data_dict['mean_cluster'],data_dict['std_cluster']).filled(0)
        cells[:,:-1]= np.ma.divide(cells[:,:-1]-data_dict['mean_cell'],data_dict['std_cell']).filled(0)

        cells = cells.reshape(clusters.shape[0],num_part,-1)

        print(f"\nL 380: Shape of Cells in DataLoader = {np.shape(cells)}") 
        # Shape of Cells in DataLoader = (69930, 200, 5)

        print(f"\nL 381: Cells in DataLoader = \n{cells[0,15:20,:]}")

        return cells.astype(np.float32),clusters.astype(np.float32)


    for label in labels:

        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['cluster'][:].shape[0]
            # ntotal = int(nevts)

            print("NTOTAL = ",ntotal)
            if make_tf_data:
                cell=h5f['hcal_cells'][rank:int(0.7*ntotal):size].astype(np.float32)
                cluster=h5f['cluster'][rank:int(0.7*ntotal):size].astype(np.float32)

            else:
                #load evaluation data
                cell = h5f['hcal_cells'][int(0.7*ntotal):].astype(np.float32)
                cluster = h5f['cluster'][int(0.7*ntotal):].astype(np.float32)

            print("#"*40)
            print("CELLs in DATALOADER = ",cell)
            print("-"*40)
            print("Clusters in DATALOADER = ",cluster)
            print("#"*40)
            cells.append(cell)
            clusters.append(cluster)

    cells = np.concatenate(cells)
    clusters = np.concatenate(clusters)

    print('clusters', clusters.shape) # clusters (69930, 4)

    #Split Cluster Data into Input and Condition
    cond = clusters[:,:num_condition]#GenP, GenTheta 

    # FIXME: PLEASE BE CAREFULE TO ONLY USE nclusters_var
    clusters = clusters[:,:] #ClusterSum, N_Hits
    # clusters = clusters[:,ncluster_var:] #ClusterSum, N_Hits

    print('cond', cond.shape) # cond (69930, 2)
    print('clusters', clusters.shape) # clusters (69930, 2) ; while plotting clusters (29970, 2)
    print('*'*30)
    print(cond[0]) # [55.11317  18.366808]
    print(clusters[0]) # [ 0.35388187 67.]

    print('mean', np.mean(cond, axis=0)) # mean [ 1.0650898 17.038385 ]
    print('min', np.min(cond, axis=0)) # min [9.4318224e-05 1.5567589e+01]
    print('max', np.max(cond, axis=0)) # ax [ 2.1198688 18.432405 ]


    #Additional Pre-Processing, Log10 of E
    # cells[:,:,0] = np.log10(cells[:,:,0]) #Log10(CellE)
    print("&"*10,np.min(cells[:,:,0]),"&"*10)

    cellsE = cells[:,:,0]
    print(cellsE)
    cells[:,:,0] = np.where(cellsE > 0.0, np.log10(cellsE), 0.0 )
    cond[:,0] = np.log10(cond[:,0]) #Log10 of GenP 
    print(cells[:,:,0])

    # clusters = np.log10(clusters[:,0]) # ClusterSumE, after cond split

    cells,clusters,cond = shuffle(cells, clusters, cond, random_state=0)
    cells,clusters = _preprocessing(cells, clusters, save_json=True) 
    # cells,clusters = _preprocessing(cells,clusters,save_json=False) 

    print('cells',cells.shape) # cells (69930, 200, 5)
    print('clusters',clusters.shape) # clusters (69930, 2)

    # Do Train/Test Split, or just return data
    data_size = clusters.shape[0]

    if make_tf_data:

        train_cells = cells[:int(0.8*data_size)] #This is 80% train (whcih 70% of total)
        train_clusters = clusters[:int(0.8*data_size)]
        train_cond = cond[:int(0.8*data_size)]

        test_cells = cells[int(0.8*data_size):]
        test_clusters = clusters[int(0.8*data_size):]
        test_cond = cond[int(0.8*data_size):]

        def _prepare_batches(cells,clusters,cond):

            nevts = clusters.shape[0]
            mask = np.expand_dims(cells[:, :, -1], -1)
            masked = cells[:, :, :-1]*mask
            masked[masked[:, :, :] == -0.0] = 0.0

            # tf_cluster = tf.data.Dataset.from_tensor_slices(clusters)
            clusters_tensor = torch.tensor(clusters)
            cond_tensor = torch.tensor(cond)
            part_tensor = torch.tensor(masked)
            mask_tensor = torch.tensor(mask)

            torch_cond = TensorDataset(cond_tensor)
            torch_part = TensorDataset(part_tensor)
            torch_mask = TensorDataset(mask_tensor)
            torch_cluster = TensorDataset(clusters_tensor)

            # Really good check on mask and data before training
            print("First Cells in _prepare_batches = \n", masked[10, :10, :])
            print("Last Cells in _prepare_batches = \n", masked[10, -10:, :])


            zipped_dataset = ZipDataSets(torch_part, torch_cluster, torch_cond, torch_mask)

            dataloader = DataLoader(
                zipped_dataset,
                batch_size=batch_size,
                shuffle=True,  # This enables shuffling
                num_workers=0,  # Adjust as per your requirement
                drop_last=True  # Drops the last incomplete batch, if the dataset size is not divisible by the batch size
            )

            return dataloader
            # return tf_zip.shuffle(nevts).repeat().batch(batch_size)

    
        train_data = _prepare_batches(train_cells,train_clusters,train_cond)
        test_data  = _prepare_batches(test_cells,test_clusters,test_cond)    
        return data_size, train_data, test_data
    
    else:
        
        #print('mean', np.mean(cond, axis=0)) # mean [ 1.0650898 17.038385 ]
        #print('min', np.min(cond, axis=0)) # min [9.4318224e-05 1.5567589e+01]
        #print('max', np.max(cond, axis=0)) # ax [ 2.1198688 18.432405 ]
        print(cond.shape) # (29970, 2)
        mask = np.expand_dims(cells[:nevts,:,-1],-1)
        print(cond[:nevts].shape) # (10, 2)
        return cells[:nevts,:,:-1]*mask,clusters[:nevts], cond[:nevts]
