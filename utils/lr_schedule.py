import numpy as np


def lr_schedule(optimizer,
                iternum,
                global_bs,
                base_bs,
                scaling='none',
                start_lr=1e-4,
                tot_steps=1000,
                end_lr=0.,
                warmup_steps=0):

    if scaling == 'sqrt':
        init_lr = np.sqrt(global_bs/base_bs)*start_lr
    elif scaling == 'linear':
        init_lr = (global_bs/base_bs)*start_lr
    elif scaling == 'none':
        init_lr = start_lr

    if global_bs > base_bs and scaling != 'none':
        # warm-up lr rate
        if iternum < warmup_steps:
            lr = (iternum/warmup_steps)*init_lr
        else:
            lr = end_lr + 0.5 * (init_lr - end_lr) * \
                (1 + np.cos(np.pi * (iternum - warmup_steps)/tot_steps))
    else:
        lr = end_lr + 0.5 * (init_lr - end_lr) *\
            (1 + np.cos(np.pi * iternum/tot_steps))

    optimizer.param_groups[0]['lr'] = lr
