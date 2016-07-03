

class Params:
    def __init__(self, epochs, train_size, test_size, corpus, hidden_nodes=list(), slice_samples=512, batch_size=1000):
        self.epochs = epochs
        self.train_size = train_size
        self.test_size = test_size
        self.corpus = corpus
        self.slice_samples = slice_samples
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "corpus": self.corpus,
            "slice_samples": self.slice_samples,
            "batch_size": self.batch_size,
            "hidden_nodes": self.hidden_nodes
        }
