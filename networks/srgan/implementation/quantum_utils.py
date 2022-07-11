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


def quanv(image):
    """
    Convolves the input image with many applications of the same quantum circuit.

    Inputs
        :image: <np.ndarray> representing the image
    
    Outputs
        :returns: <np.ndarray> of the preproccesed image
    """
    h, w, c = image.shape
    out = np.zeros((h, w, c, 4))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, h, 2):
        for k in range(0, w, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[j, k, :],
                    image[j, k + 1, :],
                    image[j + 1, k, :],
                    image[j + 1, k + 1, :]
                ]
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for i in range(c):
                out[j // 2, k // 2, i] = q_results[i]
    return out