from torch.utils.data import DataLoader

class BaseLoaderWrapper:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def stats(self):
        if isinstance(self.loader, BaseLoaderWrapper):
            return self.loader.stats()
        else:
            return {}

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.loader)

    def __repr__(self):
        return str(self)