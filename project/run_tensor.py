"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .

Disclaimer: AI Claude 3.5 Sonnet (Cursor on Mac) was used to help write comments and code for this file."""


import minitorch
import numpy as np
import time

# Use this function to make a random parameter in
# your module.
def RParam(*shape: int) -> minitorch.Parameter:
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()

        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        start = self.layer1.forward(x).relu()
        middle = self.layer2.forward(start).relu()
        return self.layer3.forward(middle).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        # Reshape inputs and weights for batch matrix multiplication
        batch_size = x.shape[0]
        x = x.view(batch_size, self.in_size, 1)
        w = self.weights.value.view(1, self.in_size, self.out_size)
        b = self.bias.value.view(1, self.out_size)

        # Compute output: (batch_size, in_size, 1) * (1, in_size, out_size) -> (batch_size, out_size)
        out = (x * w).sum(1).view(batch_size, self.out_size)
        return out + b


class TensorTrain:
    def __init__(self, hidden_layers: int):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x: list) -> minitorch.Tensor:
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X: list) -> minitorch.Tensor:
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate: float, max_epochs: int = 500,
             log_fn = lambda epoch, loss, correct, _, time_per_epoch: print(f"Epoch {epoch}, loss {loss}, correct {correct}, time per epoch {time_per_epoch}")):
        """Train the model using binary cross entropy loss."""
        # Initialize model and optimizer
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        # Convert data to tensors
        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        times = []
        for epoch in range(1, max_epochs + 1):
            # Forward pass
            start_time = time.time()
            optim.zero_grad()
            out = self.model.forward(X).view(data.N)

            # Binary cross entropy loss
            loss = -(y * out.log() + (1.0 + (-y)) * (1.0 + (-out)).log())
            loss_mean = loss.sum() / data.N



            # Backward pass
            loss_mean.backward()
            optim.step()

            # Track metrics
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            correct = int(((out.detach() > 0.5) == y).sum()[0])

            times.append(time.time() - start_time)

            # Log progress
            if epoch % 10 == 0 or epoch == max_epochs:

                log_fn(epoch, total_loss, correct, losses, np.mean(times[-10:]))


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 100
    RATE = 0.05
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
