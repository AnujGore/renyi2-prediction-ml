import numpy as np
import torch
from scipy.linalg import expm

from .utils import pauli_basis, kron_prod_row, unitary_check

def generateRandomVector(n):
    random_vec = (np.pi)*(torch.rand(4**n - 1)-0.5)

    return random_vec
    
def generateUnitary(n, rotated_q, pauli_strings, haar = False):
    if haar == True:
        theta = generateRandomVector(rotated_q)
        summed_pauli_matrices = (theta.view(-1, 1, 1) * pauli_strings).sum(dim = 0)
        unitary_prime = torch.tensor(expm((0-1j)*summed_pauli_matrices))
    elif haar == False:
        idx = torch.randint(low = 0, high = 3, size = (1,))
        theta = pauli_strings[idx].flatten()
        unitary_prime = pauli_strings[idx].squeeze(0)
    
    not_rotated_q = n - rotated_q #Doing identity over the un-rotated ones
    identity_mat_row = pauli_basis[torch.full((not_rotated_q, ), 0)]
    unitary = torch.kron(unitary_prime, kron_prod_row(identity_mat_row))

    return unitary, theta
