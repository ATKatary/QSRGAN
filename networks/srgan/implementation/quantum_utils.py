import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

### Global Constants ###
n_layers = 1
dev = qml.device("default.qubit", wires=4)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    """
    Constructs a circuit with angle phi

    Inputs
        :phi: <float> rotation angle in [0, pi]
    """
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]


def quanv(image, n):
    """
    Convolves the input image with many applications of the same quantum circuit.
    Downsamples the image by factor of k

    Inputs
        :image: <np.ndarray> representing the image
        :n: <int> scalling factor
    
    Outputs
        :returns: <np.ndarray> of the preproccesed image
    """
    h, w, c = image.shape
    out = np.zeros((h // n, w // n, c, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for i in range(0, h, n):
        for j in range(0, w, n):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[i, j, :],
                    image[i, j + 1, :],
                    image[i + 1, j, :],
                    image[i + 1, j + 1, :]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for k in range(c):
                out[i // 2, j // 2, k] = q_results[k]
    return out