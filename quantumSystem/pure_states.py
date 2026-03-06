import numpy as np
import torch
from itertools import product

class pureState:
    def __init__(self, n):
        self.n = n
        self.density_matrix = self.groundStateGenerator()
        self.computational_basis_outcomes = [i for i in product(range(2), repeat=self.n)]
        self.ket_vector = torch.zeros(size = (2**self.n, ), dtype=torch.complex64)
        self.ket_vector[0] = 1

    def groundStateGenerator(self):
        rho = torch.zeros(size=(2**self.n, 2**self.n), dtype=torch.complex64)
        rho[0][0] = 1

        return rho
   
    def reset_density_matrix(self, new_density_matrix):
        self.density_matrix = new_density_matrix

    def return_exp_val(self, expectation_val_basis):
        exp_vals = torch.zeros(size=(expectation_val_basis.size(0), ))
        for idx, basis in enumerate(expectation_val_basis):
            exp_vals[idx] = torch.trace((self.density_matrix@basis).real)
        return exp_vals

    def measure(self):
        probability_tensor = torch.abs(torch.diagonal(self.density_matrix))
        probability_vector = probability_tensor.cpu().numpy()
        outcome_idx = np.random.choice(np.arange(2**self.n), p = probability_vector)

        return torch.tensor(self.computational_basis_outcomes[outcome_idx]), outcome_idx
    
    def haarRandomize(self):
        real_part = torch.randn(2**self.n)
        imag_part = torch.randn(2**self.n)

        ket = real_part + 1j * imag_part

        norm = torch.linalg.norm(ket)

        if norm != 0:
            ket = ket / norm

        density_matrix = torch.outer(ket, ket.conj())

        self.ket_vector = ket
        self.density_matrix = density_matrix

    
    def schmidtGap(self):
        subA_dim = int(self.n / 2) 
        subB_dim = int(self.n - subA_dim)

        _, eigs, _ = np.linalg.svd(self.ket_vector.reshape(2**subA_dim, 2**subB_dim))

        self.eigs = eigs

        return np.abs(eigs[0] - eigs[1])
    
    def vonNeumann(self):
        subA_dim = int(self.n / 2) 
        subB_dim = int(self.n - subA_dim)

        _, eigs, _ = np.linalg.svd(self.ket_vector.reshape(2**subA_dim, 2**subB_dim))
        eigs = eigs[np.real(eigs) != 0]

        return -1 * np.real(sum(eigs**2 * np.log2(eigs**2)))
    
    def renyi(self):
        subA_dim = int(self.n / 2) 
        subB_dim = int(self.n - subA_dim)

        _, eigs, _ = np.linalg.svd(self.ket_vector.reshape(2**subA_dim, 2**subB_dim))
        eigs = eigs[np.real(eigs) != 0]

        return -1 * np.real(np.log2(sum(eigs**4)))

    
    def maximallyEntangled(self):
        self.ket_vector[0] = 1 / torch.sqrt(torch.tensor(2.0)) 
        self.ket_vector[-1] = 1 / torch.sqrt(torch.tensor(2.0))
        
        self.density_matrix[0, 0] = 1/2
        self.density_matrix[0, -1] = 1/2
        self.density_matrix[-1, 0] = 1/2
        self.density_matrix[-1, -1] = 1/2