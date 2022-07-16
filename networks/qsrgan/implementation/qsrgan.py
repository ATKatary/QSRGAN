import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

### Global Constants ###
q_depth = 6             # Depth of the parameterised quantum circuit / D
n_qubits = 5            # Total number of qubits / N
n_a_qubits = 1          # Number of ancillary qubits / N_A


dev = qml.device("lightning.qubit", wires=n_qubits) # Quantum simulator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QSRGAN(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators = 4, k = 2, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()
        self.scale_factor = k

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, input, mode = 'bilinear'):
        f = nn.Upsample(scale_factor=self.scale_factor, mode=mode)
        input = f(input)

        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. input.size(0) is the batch size.
        images = torch.Tensor(input.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in input:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images
    
    def initiate(self, device, pretrained_weights):
        """ Initializes the model using pretrained weights """
        print("Initializing ...")
        self.load_state_dict(pretrained_weights)
        self.eval()
        return self

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):
    """
    """
    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))

def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven
