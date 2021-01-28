class Data:
    def sample_generator(self):
        raise NotImplementedError

    def types(self):
        raise NotImplementedError

    def shapes(self):
        raise NotImplementedError

    def array_to_str(self, arr):
        raise NotImplementedError


class UnkDict:
    unk = '<UNK>'

    def __init__(self, items):
        if self.unk not in items:
            raise ValueError("items must contain %s", self.unk)

        self.delegate = dict([(c, i) for i, c in enumerate(items)])
        self.rdict = {i: c for c, i in self.delegate.items()}

    def __getitem__(self, item):
        if item in self.delegate:
            return self.delegate[item]
        else:
            return self.delegate[self.unk]

    def __len__(self):
        return len(self.delegate)

    def idx2key(self, idx):
        return self.rdict[idx]
