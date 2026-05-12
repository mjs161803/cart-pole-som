class Node:
    def __init__(self, w, idx):
        self.index = idx
        self.weights = w
        self.accumulated_error = 0.0
        self.accumulated_age = 0
