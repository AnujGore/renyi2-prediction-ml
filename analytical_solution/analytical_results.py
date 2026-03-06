import numpy as np
from itertools import product
from rich.progress import track

import matplotlib.pyplot as plt

from scipy.stats import unitary_group

import os
import sys
sys.path.append('.') 
from quantumSystem.pure_states import pureState


def generate_tensor_unitaries(n):
    """
    :params int n: number of qubits on the bipartition (must be smaller than the total number of qubits)
    """

    u = unitary_group.rvs(2)

    for _ in range(1, n):
        u = np.kron(u, unitary_group.rvs(2))

    return u

def reduced_density_matrix(rho):
    subsystem_size = int(np.log2(rho.shape[0])/2)
    dA, dB = 2**subsystem_size, 2**(int(np.log2(rho.shape[0])-subsystem_size))
    rho_reshaped = rho.reshape(dA, dB, dA, dB)
 
    rho_red = np.zeros((dA, dA), dtype=complex)

    for i in range(dB):
        idx = np.ix_(range(dA), [i], range(dA), [i])
        rho_red += rho_reshaped[idx].squeeze()
   
    return rho_red

def calculate_hamming_matrix(n):
    hamming_matrix = np.zeros(shape = (2**n, 2**n), dtype=np.int64)
    logical_basis_states = [i for i in product(range(2), repeat=n)]
    for a in range(2**n):
        for b in range(2**n):
            a_bit = logical_basis_states[a]
            b_bit = logical_basis_states[b]

            hamming_distance = sum(np.bitwise_xor(a_bit, b_bit))

            hamming_matrix[a, b] = hamming_distance
    
    return hamming_matrix


def brydges_formula(system: pureState, nu, nm):
    results = np.zeros(shape=(nu, nm, 2**int(system.n/2)))
    n_all = system.n

    for u in range(nu):
        #Generate random unitary
        random_u = generate_tensor_unitaries(int(n_all/2))
        rdm = reduced_density_matrix(system.density_matrix.numpy())
        rotated_system = random_u@rdm@random_u.conj().T

        prob_vec = np.diag(rotated_system)
        prob_vec = prob_vec.real/np.sum(prob_vec.real)

        for m in range(nm):
            outcome = np.random.choice(np.arange(2**int(n_all/2)), p = prob_vec)
            results[u, m, outcome] += 1


    return results


if __name__ == "__main__":
    n = 4
    output_base = f"analytical_solution/results/{n}q"
    os.makedirs(output_base, exist_ok=True)

    coeff_matrix = np.power(-0.5, calculate_hamming_matrix(int(n/2)), dtype=np.float64)

    nu_max = 500
    nm_max = 200
    n_samples = 1000

    trues = []

    preds = np.zeros(shape = (n_samples, nu_max, nm_max, 2**int(n/2)))

    for s in track(range(n_samples), description="Generating results"):
        this_system = pureState(n)
        this_system.haarRandomize()

        results = brydges_formula(this_system, nu_max, nm_max)

        trues.append(this_system.renyi())

        preds[s] = results

    shots_list = np.logspace(np.log10(1), stop=np.log10(nm_max), num=10, dtype=int)
    unitaries_list = np.linspace(1, stop=nu_max, num=10, dtype=int)

    loss = []

    for i, u in track(enumerate(unitaries_list), description="Evaluating over Unitaries"):
        for j, shot in enumerate(shots_list):
            preds_spliced = []
            for t in range(n_samples):
                this_results = preds[t]
                spliced_results = this_results[:u, :shot, :]
                prob_vec = spliced_results.mean(axis = 1)
                purities = []
                for r in range(prob_vec.shape[0]):
                    purity = (2**(int(n/2)))*np.sum(np.multiply(coeff_matrix, np.outer(prob_vec[r], prob_vec[r])))
                    purities.append(purity)
                
                preds_spliced.append(-np.log2(np.mean(purities)))

            diff = np.abs(np.array(preds_spliced) - np.array(trues))

            loss.append([shot, u, np.mean(diff), np.std(diff)])

    
    np.savetxt(f"{output_base}/testing_loss.txt", loss)