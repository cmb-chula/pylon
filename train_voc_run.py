from train_voc import *

if __name__ == "__main__":
    confs = []
    for seed in [0]:
        confs += voc_baseline(seed, 256)
        confs += voc_li2018(seed, 256)
        confs += voc_fpn(seed, 256)
        confs += voc_pylon_two_phase(seed, 256)

    debug = False
    multiprocess_map(
        Run(
            namespace='',
            debug=debug,
            train=True,
            test_auc=True,
            test_loc=True,
            gen_picked=True,
        ),
        confs,
        num_workers=len(confs),
        progress=True,
        debug=debug,
    )
