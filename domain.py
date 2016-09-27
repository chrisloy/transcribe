

class Params:
    def __init__(self, epochs, train_size, test_size, corpus, batch_size, **kwargs):
        self.epochs = epochs
        self.train_size = train_size
        self.test_size = test_size
        self.corpus = corpus
        self.batch_size = batch_size
        self.lower = 21
        self.upper = 109
        self.padding = 0
        self.hidden_nodes = list()
        self.slice_samples = 512
        self.graph_type = "mlp"
        self.learning_rate = 0.001
        self.steps = None
        self.hidden = None
        self.dropout = False
        self.subsample = None
        self.batch_norm = False
        self.__dict__.update(kwargs)

    def outputs(self):
        return self.upper - self.lower
