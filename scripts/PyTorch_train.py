import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import utils
import sys
from point_cloud_diffusion import PCD
from GSGM_distill import GSGM_distill
import time

torch.manual_seed(1233)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_cluster.json',
                        help='Config file with training parameters')

    parser.add_argument('--data_path', default='./',
                        help='Path containing the training files')

    parser.add_argument('--distill', action='store_true',
                        default=False, help='Use the distillation model')

    parser.add_argument('--big', action='store_true', default=False,
                        help='Use bigger dataset (1000 particles) \
                        as opposed to 200 particles')

    parser.add_argument('--factor', type=int, default=1,
                        help='Step reduction for distillation model')

    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for DDP')

    flags = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(flags.local_rank)

    config = utils.LoadJson(flags.config)

    assert flags.factor % 2 == 0 or flags.factor == 1,\
    "Distillation reduction steps needs to be even"

    if flags.big:
        labels = utils.labels1000
        npart = 1000
    else:
        labels = utils.labels200
        npart = 200

    data_size, training_data, test_data = utils.DataLoader(flags.data_path,
                                                           labels,
                                                           npart,
                                                           dist.get_rank(),
                                                           dist.get_world_size(),
                                                           config['NUM_CLUS'],
                                                           config['NUM_COND'],
                                                           config['BATCH'])

    model = PCD(config=config, npart=npart).cuda()

    model_name = config['MODEL_NAME']
    if flags.big:
        model_name += '_big'
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)

    # if flags.distill:
    #     if flags.factor > 2:
    #         checkpoint_folder = \
    #         '../checkpoints_{}_{}/checkpoint'.format(model_name,flags.factor // 2)

    #         model = GSGM_distill(model.ema_jet, model.ema_part,
    #                              factor=flags.factor // 2, config=config)

    #         model.load_state_dict(torch.load(checkpoint_folder))

    #         model = GSGM_distill(model.ema_jet, model.ema_part,
    #                              factor=flags.factor, config=config, npart=npart)
    #     else:
    #         model.load_state_dict(torch.load(checkpoint_folder))
    #         model = GSGM_distill(model.ema_jet, model.ema_part,
    #                              factor=flags.factor, config=config, npart=npart)

    #     if dist.get_rank() == 0:
    #         print(f"Loading Teacher from: {checkpoint_folder}")

    #     checkpoint_folder = \
    #     f'../checkpoints_{model_name}_d{flags.factor}/checkpoint'

    model = DDP(model, device_ids=[flags.local_rank])

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim.Adamax(model.parameters(), lr=config['LR']),
        T_max=config['MAXEPOCH'] * int(data_size * 0.8 / config['BATCH'])
    )

    loss_fn = nn.MSELoss()
    # FIXME: Do the below instead. You'll have to finish the GSGM Class tomorrow!!!
    # loss = model.custom_loss(outputs, targets)

    z = tf.random.normal((tf.shape(part)),dtype=tf.float32)*mask
    perturbed_x = alpha_reshape*part + z * sigma_reshape
    pred = self.model_part([perturbed_x*mask, random_t,jet,cond,mask])
    v = alpha_reshape * z - sigma_reshape * part
    losses = tf.square(pred - v)*mask
    loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

    start_time = time.time()

    # Optimizer
    optimizer = optim.Adamax(model.parameters(), lr=config['LR'])

    # Assuming training_data is a DataLoader
    sampler = DistributedSampler(training_data.dataset,
                                 num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank())

    training_data.sampler = sampler

    for epoch in range(config['MAXEPOCH']):
        model.train()
        sampler.set_epoch(epoch)
        for data, target in training_data:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()

        if dist.get_rank() == 0 and epoch % 1 == 0:
            torch.save(model.state_dict(), checkpoint_folder)

    print("--- %s seconds ---" % (time.time() - start_time))
