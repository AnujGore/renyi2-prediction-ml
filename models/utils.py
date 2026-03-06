import torch
from torch.utils.data import Dataset
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import numpy as np
import os


#Dataset organization    
class SystemDataset(Dataset):
    def __init__(self, r_list, o_list, s_list):
        """
        r_list, o_list: list of [n_i, r_dim] / [n_i, o_dim] tensors for each system
        s_list: list of [1] tensors (ground truth scalars)
        """
        self.r_list = r_list
        self.o_list = o_list
        self.s_list = s_list

    def __len__(self):
        return len(self.r_list)

    def __getitem__(self, idx):
        r = self.r_list[idx]
        o = self.o_list[idx]
        s = self.s_list[idx]

        # Random permutation of the system's rows
        perm = torch.randperm(r.size(0))
        r_shuffled = r[perm]
        o_shuffled = o[perm]

        return r_shuffled, o_shuffled, s



#Stella architecture
class MutableTrainState(train_state.TrainState):
    mutable_state: dict

def create_state_attention(rng, model, unitary_shape, outcome_shape, learning_rate, device=None):
    state = model.init(rng, jnp.ones(unitary_shape), jnp.ones(outcome_shape))

    params = state["params"]
    mutable_state = {"state": state["state"]}

    tx = optax.chain(
        optax.adam(learning_rate)
    )

    if device is not None:
        params = jax.device_put(params, device)
        mutable_state = {k: jax.device_put(v, device) for k, v in mutable_state.items()}

    return MutableTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mutable_state=mutable_state
    )

@jax.jit
def train_step_attention(state, batch):
    def loss_fn(params):
        preds, updated_state = state.apply_fn({"params": params, **state.mutable_state}, batch["unitaries"], batch["outcomes"], compute_attn = True, mutable = ["state"])
        preds = jnp.squeeze(preds.real)
        loss = optax.l2_loss(preds, batch["renyi"]).mean()
        return loss, updated_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updated_state), grads = grad_fn(state.params)
   
    state = state.apply_gradients(grads = grads)

    state = state.replace(mutable_state = updated_state)

    return state

@jax.jit
def eval_step_attention(state, batch):
    preds = state.apply_fn({"params": state.params, **state.mutable_state}, batch["unitaries"], batch["outcomes"], compute_attn = False, mutable = False)
    preds = preds.real.squeeze(-1)
    loss = optax.l2_loss(preds, batch["renyi"]).mean()
    return {"loss": loss,
            "preds": preds,
            "target": batch["renyi"]}



#MLP architecture
def create_state_UnitaryMLP(rng, model, learning_rate, unitary_shape, outcome_shape, num_epochs, device=None):
    params = model.init(rng, jnp.ones(unitary_shape), jnp.ones(outcome_shape))["params"]
    schedule = optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales={50: 0.1, 100: 0.1}
        )

    tx = optax.chain(
        optax.adam(schedule)
    )

    if device is not None:
        params = jax.device_put(params, device)

    return train_state.TrainState.create(apply_fn=model.apply, params = params, tx = tx)

@jax.jit
def train_step_UnitaryMLP(state, batch):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch["unitaries"], batch["outcomes"])
        preds = jnp.squeeze(preds.real)
        loss = optax.l2_loss(preds, batch["renyi"]).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
   
    state = state.apply_gradients(grads = grads)

    return state

@jax.jit
def eval_step_UnitaryMLP(state, batch):
    preds = state.apply_fn({"params": state.params}, batch["unitaries"], batch["outcomes"])
    preds = preds.real.squeeze(-1)
    loss = optax.l2_loss(preds, batch["renyi"]).mean()
    return {"loss": loss,
            "preds": preds,
            "target": batch["renyi"]}


def analyze_prediction_vs_true(pred, true):
    # Linear regression: fit line y = a * x + b
    a, b = np.polyfit(true, pred, 1)

    theta_rad = np.arctan((a - 1) / (1 + a))  # angle between slopes
    angle_deg = np.degrees(theta_rad)


    return angle_deg


#Loading old(new) dataset
def load_or_generate_dataset(n, n_systems, n_copies, n_shots, generateDataset, base_dir):
    base_path = f"{base_dir}/{n}qubits_{n_copies}copies_{n_shots}shots"
    thetas_file = f"{base_path}_thetas.pt"
    outcomes_file = f"{base_path}_outcomes.pt"
    entropy_file = f"{base_path}_entropy.pt"
    parity_file = f"{base_path}_parity.pt"
    unitaries_file = f"{base_path}_unitaries.pt"
    probs_file = f"{base_path}_probs.pt"
    renyi_file = f"{base_path}_renyi.pt"

    # Check if all required files exist
    if all(os.path.exists(f) for f in [thetas_file, outcomes_file, entropy_file, parity_file, unitaries_file, probs_file, renyi_file]):
        thetas = torch.load(thetas_file)
        outcomes = torch.load(outcomes_file)
        entropy = torch.load(entropy_file)
        parity = torch.load(parity_file)
        unitaries = torch.load(unitaries_file)
        probs = torch.load(probs_file)
        renyi = torch.load(renyi_file)
    else:
        # Generate the dataset
        thetas, outcomes, entropy, parity, unitaries, probs, renyi = generateDataset(
            n, n_systems, n_copies, haar=True, shots=n_shots
        )

        # Save all files
        for data, file in zip(
            [thetas, outcomes, entropy, parity, unitaries, probs, renyi],
            [thetas_file, outcomes_file, entropy_file, parity_file, unitaries_file, probs_file, renyi_file]
        ):
            os.makedirs(os.path.dirname(file), exist_ok=True)
            torch.save(data, file)

    return thetas, outcomes, entropy, parity, unitaries, probs, renyi


