from trainer.start import *


class BaseCombinedDataset:
    def make_loader(self, dataset, shuffle, **kwargs):
        return DataLoader(
            dataset,
            batch_size=self.conf.bs,
            num_workers=self.conf.n_worker,
            shuffle=shuffle,
            multiprocessing_context=(mp.get_context('fork')
                                     if self.conf.n_worker > 0 else None),
            collate_fn=bbox_collate_fn,
            **kwargs,
        )


def bbox_collate_fn(data):
    out = defaultdict(list)
    keys = ['img', 'classification', 'bboxes']
    for each in data:
        out['img'].append(each['img'])
        out['classification'].append(each['classification'])
        if 'bboxes' in each:
            out['bboxes'].append(each['bboxes'])
        else:
            out['bboxes'].append([])
        # include other keys
        for k, v in each.items():
            if k not in keys:
                out[k].append(v)
    out['img'] = torch.stack(out['img'])
    out['classification'] = torch.stack(out['classification'])
    return out