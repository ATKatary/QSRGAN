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

    def __init__(self, in_c = 3, k = 2, n_qubits = 4, n_a_qubits = 0):
        """
        AF(in_c, k, n_qubits, n_a_qubits) = a single layer network (a single ciruit network) with n_qubits, upscaling the image by a factor of k
                                            with n_a_qubits acting as ancillary qubits (supporting)

        Representation Invariant
            - true
            - inherits from nn.Module

        Representation Exposure
            - safe
            - inherts from nn.Module
        """

        super().__init__()
        self.scale_factor = k               # upscaling factor
        self.n_qubits = n_qubits            # total number of qubits / N (must be a perfect square number)
        self.n_a_qubits = n_a_qubits        # number of ancillary qubits / N_A (aka support qubits)

        self.conv1_weights = nn.Parameter(torch.rand((in_c, self.n_qubits)), requires_grad=True)

    def forward(self, input, mode = 'bilinear'):
        f = nn.Upsample(scale_factor=self.scale_factor, mode=mode)
        input = f(input)

        output = self._quanv_layer(input, 2, 1, self.conv1_weights, self.n_qubits, self.n_a_qubits)

        return output.to(device)
    
    def initiate(self, device, pretrained_weights):
        """ Initializes the model using pretrained weights """
        print("Initializing ...")
        self.load_state_dict(pretrained_weights)
        self.eval()
        return self

    def _quanv_layer(self, images, kernel_size, stride, weights, n_qubits, n_a_qubits = 0):
        """
        Convolves the input image with many applications of the same quantum circuit.
        Downsamples the image by factor of k

        Inputs
            :images: <np.ndarray> representing the image of size b x c x h x w
            :kernel_size: <int> size of the kernel
            :stride: <int> size of the stride
            :weights: <torch.Tensor> representing the kernel 
            :n_qubits: <int> number of qubits
            :n_a_qubits: <int> 
        
        Outputs
            :returns: <np.ndarray> of the preproccesed image of size h / n x h / w x c x 4
        """
        print("\nQuanvolving ...")

        result = torch.empty(images.shape)
        for b_i in range(len(images)):
            image = images[b_i]
            c, h, w = image.shape
            out = torch.Tensor(np.zeros((h // stride, w // stride, c)))

            for i in range(0, h, stride):
                for j in range(0, w, stride):
                    q_results = quantum_circuit(self._identity(image.permute(1, 2, 0), i, j, kernel_size), weights, n_qubits)

                    for k in range(c): 
                        probs_given0 = q_results[k][:2 ** (n_qubits - n_a_qubits)]
                        probs_given0 /= torch.sum(probs_given0) 
                        
                        out[i // stride, j // stride, k] = torch.max(probs_given0)
            
            result[b_i] = out.permute(2, 0, 1)

        return result

    def _identity(self, A, i, j, k):
        """
        Returns the matrix multiplication of I_k x A[i : i + k][j : j + k][::] where I_k is the k x k identity matrix

        Inputs
            :A: <np.ndarray | torch.Tensor> representing the matrix
            :i: <int> of dim 1 coordinate
            :j: <int> of dim 2 coordinate
            :k: <int> indentiy matrix size
        
        Output
            :returns: k x k submatrix of A starting at A[i][j] in top left corner
        """
        A_k = []
        n, m = A.shape[:2]
        for x in range(k):
            x += i
            if x >= n: x = n - 1
            for y in range(k):
                y += j
                if y >= m: y = m - 1
                A_k.append(A[x, y, :])

        return A_k

@qml.qnode(dev, interface = "torch", diff_method = "parameter-shift")
def quantum_circuit(input, weights, n_qubits):
    """
    A basic quantum circuit of 4 qubits
    q1 - Ry -- Ry -- . ------------ prob1
    q2 - Ry -- Ry -- Z -- . ------- prob2
    q3 - Ry -- Ry ------- Z -- . -- prob3
    q4 - Ry -- Ry ------------ Z -- prob1

    Inputs
        :input: <list> of values of the four qubits to be passed through the circuit 
        :weights: <torch.Tensor> of trainable weights
        :n_qubits: <int> number of qubits the circuit takes in 
    
    Outputs
        :returns: <torch.Tensor> of the qubit probabilities
    """
    c, q = weights.shape

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(input[i], wires=i)

    # Repeated layer
    for i in range(c):
        # Parameterised layer
        for y in range(q):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))

