from train_vin import *

if __name__ == "__main__":
    confs = []

    # main experiments
    # for seed in [0]:
    #     confs += vin_baseline(seed, 256)
    #     confs += vin_li2018(seed, 256)
    #     confs += vin_fpn(seed, 256)
    #     confs += vin_pylon(seed, 256)

    # transfer learning
    # for seed in [0]:
    #     confs += vin_baseline_transfer(seed, 256)
    #     confs += vin_pylon_transfer(seed, 256)
    #     confs += vin_pylon_transfer_two_phase(seed, 256)

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
