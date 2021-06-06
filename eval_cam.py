import pickle

from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM, GradCAM,
                              GradCAMPlusPlus, ScoreCAM, XGradCAM)
from tqdm.autonotebook import tqdm

from train_voc import *
from utils.localization import *


def eval_cam(seed, cam_mode):
    conf = voc_baseline(seed=seed, size=256, bs=64)[0]
    print(conf.name)

    exp = conf.make_experiment()
    trainer = exp.load_trainer()

    exp.data.conf = exp.data.conf.clone()
    exp.data.conf.bs = 1
    val_loader = exp.data.make_loader(exp.data.val, shuffle=False)

    class WrapperModel(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            out = self.net.forward(img=x)
            return out.pred

    cls_scores = defaultdict(list)

    model = WrapperModel(trainer.net)
    model.eval()
    target_layer = model.net.net.layer4[-1]

    if cam_mode == 'gradcam':
        cam = GradCAM(model=model,
                      target_layer=target_layer,
                      use_cuda='cuda:0')
    elif cam_mode == 'gradcamplusplus':
        cam = GradCAMPlusPlus(model=model,
                              target_layer=target_layer,
                              use_cuda='cuda:0')
    elif cam_mode == 'scorecam':
        cam = ScoreCAM(model=model,
                       target_layer=target_layer,
                       use_cuda='cuda:0')
    elif cam_mode == 'ablationcam':
        cam = AblationCAM(model=model,
                          target_layer=target_layer,
                          use_cuda='cuda:0')
    elif cam_mode == 'xgradcam':
        cam = XGradCAM(model=model,
                       target_layer=target_layer,
                       use_cuda='cuda:0')
    elif cam_mode == 'eigencam':
        cam = EigenCAM(model=model,
                       target_layer=target_layer,
                       use_cuda='cuda:0')
    elif cam_mode == 'eigengradcam':
        cam = EigenGradCAM(model=model,
                           target_layer=target_layer,
                           use_cuda='cuda:0')
    else:
        raise NotImplementedError

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        input_tensor = batch['img']

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        _, cls_ids = torch.nonzero(batch['classification'], as_tuple=True)
        # iterate over the present classes
        for cls_i in cls_ids:
            target_category = cls_i.item()

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            heatmap = cam(input_tensor=input_tensor,
                          target_category=target_category)

            # In this example grayscale_cam has only one image in the batch:
            heatmap = heatmap[0, :]
            row, col = argmax_2d_single(torch.from_numpy(heatmap))
            # bboxes of this class
            bboxes = []
            for each in batch['bboxes'][0]:
                if each[4] == cls_i:
                    bboxes.append(each)
            score = point_loc_acc_by_img(row, col, bboxes)
            cls_scores[target_category].append(score)

        # if i > 10:
        #     break

    results = {}
    total = 0
    support = 0
    macro = []
    for k, v in cls_scores.items():
        total += sum(v)
        support += len(v)
        score = sum(v) / len(v)
        macro.append(score)
        results[f'point_acc_{exp.data.train.id_to_cls[k]}'] = score
    results['point_acc_weighted'] = total / support
    results['point_acc_macro'] = np.array(macro).mean()
    print(results)

    path = f'eval_loc/{cam_mode}/{conf.name}.csv'
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    df = DataFrame([results])
    df.to_csv(f'eval_loc/{cam_mode}/{conf.name}.csv', index=False)
    with open(f'eval_loc/{cam_mode}/{conf.name}.pkl', 'wb') as f:
        pickle.dump(cls_scores, f)


with global_queue():
    for seed in [
            0,
            # 1,
            # 2,
            # 3,
            # 4,
    ]:
        for mode in [
                'gradcam',
                'gradcamplusplus',
                'xgradcam',
                'eigencam',
                # 'scorecam',
        ]:
            eval_cam(seed, mode)
