import torch
import itertools as it

pauli_basis = torch.zeros(size=(4, 2, 2), dtype=torch.complex64)

#identity
pauli_basis[0][0, 0] = 1
pauli_basis[0][1, 1] = 1

#pauli-x
pauli_basis[1][0, 1] = 1
pauli_basis[1][1, 0] = 1

#pauli-y
pauli_basis[2][0, 1] = 0.-1.j
pauli_basis[2][1, 0] = 0.+1.j

#pauli-z
pauli_basis[3][0, 0] = 1
pauli_basis[3][1, 1] = -1

def kron_prod_row(row):
    return_arr = row[0]
    for s in row[1:]:
        return_arr = torch.kron(return_arr, s)
    
    return return_arr

def generatePauliStringSet(n, pauli_basis, complete = False):
    
    pauli_string_comb = list(it.product([0, 1, 2, 3], repeat=n))

    if complete == False:
        pauli_string_comb = pauli_string_comb[1:]

    pauli_string_matrices = pauli_basis[torch.tensor(pauli_string_comb)]

    pauli_string_matrices = torch.stack([kron_prod_row(i) for i in pauli_string_matrices], dim = 0)

    return pauli_string_matrices

def unitary_check(U: torch.tensor):
    return torch.allclose(U @ U.conj().T, torch.eye(U.shape[0], dtype=torch.complex64), atol=1e-6)