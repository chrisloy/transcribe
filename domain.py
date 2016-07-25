

class Params:
    def __init__(self, epochs, train_size, test_size, corpus, hidden_nodes=list(), slice_samples=512, batch_size=1000,
                 graph_type="mlp"):
        self.epochs = epochs
        self.train_size = train_size
        self.test_size = test_size
        self.corpus = corpus
        self.slice_samples = slice_samples
        self.batch_size = batch_size
        self.hidden_nodes = hidden_nodes
        self.graph_type = graph_type

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "corpus": self.corpus,
            "slice_samples": self.slice_samples,
            "batch_size": self.batch_size,
            "hidden_nodes": self.hidden_nodes,
            "graph_type": self.graph_type
        }


def from_dict(dx):
    return Params(
        dx["epochs"],
        dx["train_size"],
        dx["test_size"],
        dx["corpus"],
        dx["hidden_nodes"],
        dx["slice_samples"],
        dx["batch_size"],
        dx["graph_type"]
    )
