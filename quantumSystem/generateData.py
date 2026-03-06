import torch

from .pure_states import pureState
from .unitaries import generateUnitary
from .utils import generatePauliStringSet, pauli_basis
import numpy as np
from rich.progress import track

def generateDataset(n, p, s, haar: bool, shots = 1):
    """
    Args:
        n (int): Number of qubits per quantum state.
        p (int): Number of random pure states.
        s (int): Number of measurement shadows per state.

    Returns:
        thetas (torch.Tensor): Rotation angles (half-angles).
        outcomes (torch.Tensor): Measurement outcomes (computational basis).
        entropy (torch.Tensor): Target entropy values.
    """


    pauliStringSet = generatePauliStringSet(int(n/2), pauli_basis)

    if haar:
        thetas = torch.zeros(size=(p, s, int(4**(n/2) - 1)))
    else:
        thetas = torch.zeros(size=(p, s, 2**n), dtype = torch.complex64)

    unitaries = torch.zeros(size=(p, s, (2**n)**2), dtype = torch.complex64)
    outcomes = torch.zeros(size=(p, s, n))
    probs = torch.zeros(size=(p, s, shots, 2**n))
    parity = torch.zeros(size=(p, s, 1))
    entropy = torch.zeros(size = (p, ))
    renyi = torch.zeros(size = (p, ))

    rotated_q = np.floor(n/2).astype(int) #Randomly rotating half the qubits

    for j in track(range(p), description=f"Generating Dataset for {n} qubits"):

        if haar:
            theta_temp = torch.zeros(size=(s, int(4**(n/2) - 1)))
        else:
            theta_temp = torch.zeros(size=(s, 2**n), dtype = torch.complex64)
        
        unitary_temp = torch.zeros(size=(s, (2**n)**2), dtype = torch.complex64)
        outcomes_temp = torch.zeros(size=(s, n))
        probs_temp = torch.zeros(size=(s, shots, 2**n))
        parity_temp = torch.zeros(size=(s, 1))

        this_system = pureState(n)
        this_system.haarRandomize()

        entropy[j] = torch.tensor(this_system.vonNeumann(), dtype = torch.float32)
        renyi[j] = torch.tensor(this_system.renyi())

        for i in range(s):
            unitary, theta = generateUnitary(n, rotated_q, pauliStringSet, haar)
            theta_temp[i] = theta
            unitary_temp[i] = unitary.flatten()

            rho = this_system.density_matrix  
            rho_prime = torch.matmul(torch.matmul(unitary, rho), torch.conj(unitary.T))
            this_system.reset_density_matrix(rho_prime)

            outcome, idx = this_system.measure()
            outcomes_temp[i] = outcome

            parity_outs_cumulative = torch.zeros(size = (2,))

            for shot in range(shots):
                if outcome[0] == 0:
                    parity_outs_cumulative[0] += 1
                else:
                    parity_outs_cumulative[1] += 1
                
                outcome, idx = this_system.measure()
                probs_temp[i][shot][idx] = 1

            
            parity_temp[i] = (parity_outs_cumulative[0]  - parity_outs_cumulative[1])/shots

            this_system.reset_density_matrix(rho)
        
        thetas[j] = theta_temp
        outcomes[j] = outcomes_temp
        probs[j] = probs_temp
        parity[j] = parity_temp
        unitaries[j] = unitary_temp
        
    return thetas, outcomes, entropy, parity, unitaries, probs, renyi