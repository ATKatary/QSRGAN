import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

### Global Constants ###
n_qubits = 5
dev = qml.device("lightning.qubit", wires=n_qubits) # Quantum simulator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QSRGAN(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators = 4, k = 2, q_delta = 1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()
        self.q_depth = 6             # Depth of the parameterised quantum circuit / D
        self.n_qubits = 5            # Total number of qubits / N
        self.n_a_qubits = 1          # Number of ancillary qubits / N_A
        self.scale_factor = k

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(self.q_depth * self.n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, input, mode = 'bilinear'):
        f = nn.Upsample(scale_factor=self.scale_factor, mode=mode)
        input = f(input)

        # Iterate over all sub-generators
        for params in self.q_params:
            input = partial_measure(input, 1, params, self.n_qubits).float().unsqueeze(0)

        return input
    
    def initiate(self, device, pretrained_weights):
        """ Initializes the model using pretrained weights """
        print("Initializing ...")
        self.load_state_dict(pretrained_weights)
        self.eval()
        return self

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights, n_qubits, q_depth):
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

def partial_measure(input, n, weights, n_qubits, q_depth, n_a_qubits):
    # Non-linear Transform
    probs = quanv_layer(input, n, weights, n_qubits, q_depth)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven

def quanv_layer(image, n, weights, n_qubits, q_depth):
    """
    Convolves the input image with many applications of the same quantum circuit.
    Downsamples the image by factor of k

    Inputs
        :image: <np.ndarray> representing the image of size h x w x c
        :n: <int> size of the kernel
    
    Outputs
        :returns: <np.ndarray> of the preproccesed image of size h / n x h / w x c x 4
    """
    h, w, c = image.shape
    out = np.zeros((h // n, w // n, c, n_qubits))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for i in range(0, h, n):
        for j in range(0, w, n):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = quantum_circuit(
                [
                    image[i, j, :],
                    image[i, j + 1, :],
                    image[i + 1, j, :],
                    image[i + 1, j + 1, :]
                ], weights, n_qubits, q_depth
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for k in range(c):
                out[i // 2, j // 2, k] = q_results[k]
    return out
