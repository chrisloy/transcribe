

class Params:
    def __init__(self, **kwargs):
        self.epochs = None
        self.train_size = None
        self.test_size = None
        self.corpus = None
        self.batch_size = None
        self.features = 660
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
        self.frame_epochs = None,
        self.frame_hidden_nodes = None,
        self.frame_dropout = None,
        self.frame_learning_rate = None,
        self.sequence_hidden_nodes = None,
        self.sequence_dropout = None,
        self.sequence_learning_rate = None
        self.rnn_state_size = None
        self.rnn_graph_type = None
        self.noise_var = None
        self.noise_costs = None
        self.__dict__.update(kwargs)
        self.notes = self.outputs()

    def outputs(self):
        return self.upper - self.lower
