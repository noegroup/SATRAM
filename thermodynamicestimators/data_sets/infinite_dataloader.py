import torch


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """ Dataloader that does not reset every epoch, considerable decreasing time
    spent on enumeration.

    Taken from https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()


    def __len__(self):
        return len(self.batch_sampler.sampler)


    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """


    def __init__(self, sampler):
        self.sampler = sampler


    def __iter__(self):
        while True:
            yield from iter(self.sampler)

