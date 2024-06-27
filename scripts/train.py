import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import utils
import data_loading
from data_loading import get_data_loader
import time
import sys
sys.path.insert(1, '../models/')
from point_cloud_diffusion import PCD
from yaml_loader import YParams
# from GSGM_distill import GSGM_distill

torch.manual_seed(1233)

if __name__ == "__main__":

    # Flags for running env vars (e.g. rank)
    # Params for configuration options

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/default_config.yaml',
                        help='Config file with training parameters')

    flags = parser.parse_args()

    config = utils.LoadJson(flags.config)

    # Model, Training, and Input Parameters
    params = YParams(flags.config)

    assert params.distill_factor % 2 == 0 or params.model.distill_factor == 1,\
    "Distillation reduction steps needs to be even"

    npart = params.n_part_max

    if npart == 200:
        labels = utils.labels200

    elif npart == 1000:
        labels = utils.labels1000

    world_rank = 0
    local_rank = 0

    if params.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_batch_size:
      # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size" : world_size*args.local_batch_size})
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = params.global_batch_size//world_size

    train_loader, val_loader = get_data_loader(params, world_rank, device=0)

    # data_size, training_data, test_data = data_loading.MainDataLoader(flags.data_path,
    #                                                        labels,
    #                                                        npart,
    #                                                        # dist.get_rank(),
    #                                                        # dist.get_world_size(),
    #                                                        config['NUM_CLUS'],
    #                                                        config['NUM_COND'],
    #                                                        config['BATCH'])

    model = PCD(config_file=flags.config, npart=npart).cuda()
    model_name = config['MODEL_NAME']

    if params.big:
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

    start_time = time.time()


    # Optimizer
    optimizer = optim.Adamax(model.parameters(), lr=config['LR'])

    # Assuming training_data is a DataLoader
    sampler = DistributedSampler(training_data.dataset,
                                 num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank())

    training_data.sampler = sampler

    # for epoch in range(num_epochs):
    #     for batch in train_loader:
    #         particles, jets, conditions, mask = batch
    #         print(particles, jets, conditions, mask)

    # FIXME: adapt from train.py
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
