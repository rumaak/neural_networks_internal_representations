import torch

class Dataset:
    def __init__(self, name, device, used_by="notebook"):
        self.used_by = used_by
        self.device = device

        if name == "simple":
            self.load_simple()
        else:
            print("Unknown dataset")

    def load_simple(self):
        self.size = 9
        self.dim = 25
        self.shape = (5,5)
        self.n_classes = 3

        self.class_to_index = {1:0, 3:1, 4:2}
        self.index_to_class = [1, 3, 4]

        if self.used_by == "notebook":
            filename = "../data/simple"
        else:
            # TODO
            filename = "../data/simple"

        self.labels = torch.ones(self.size, dtype=torch.long).to(self.device)
        self.data = torch.ones(self.size, self.dim, dtype=torch.float).to(self.device)

        with open(filename) as f:
            for i,line in enumerate(f):
                parts = line.split()
                self.labels[i] = self.class_to_index[int(parts[0])]
                for j,n in enumerate(parts[1]):
                    self.data[i,j] = int(n)

