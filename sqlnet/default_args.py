class Args(object):
    def __init__(self, **kwargs):
        self.toy = False
        self.suffix = ''
        self.ca = False
        self.dataset = 0
        self.rl = False
        self.baseline = False
        self.train_emb = False
        for k, v in kwargs.items():
            setattr(self, k, v)
