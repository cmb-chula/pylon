from train_nih import *

if __name__ == "__main__":
    confs = []

    # main experiments
    for seed in [0]:
        confs += nih_baseline(seed, 256)
        confs += nih_li2018(seed, 256)
        confs += nih_unet(seed, 256)
        confs += nih_fpn(seed, 256)
        confs += nih_deeplabv3(seed, 256)
        confs += nih_pan(seed, 256)
        confs += nih_pylon(seed, 256)

    # 512 x 512 experiments
    for seed in [0]:
        confs += nih_baseline(seed, 512)
        confs += nih_li2018(seed, 512)
        confs += nih_fpn(seed, 512)
        confs += nih_pylon(seed, 512)

    debug = False
    multiprocess_map(
        Run(
            namespace='',
            debug=debug,
            train=True,
            test_auc=True,
            test_loc=True,
        ),
        confs,
        num_workers=len(confs),
        progress=True,
        debug=debug,
    )
