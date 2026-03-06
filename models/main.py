from models.networks import *
from models.utils import *
from models.testing import test_model
from quantumSystem.generateData import generateDataset

from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import sys
from rich.console import Console
from rich.progress import track
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
num_epochs = 5000
learning_rate_MLP = 1e-5
learning_rate_stella = 5e-8
eval_after = 10
repeats_per_shot = 5

embedding_hidden_dim_2q = [128, 512, 128]; hidden_layers_2q = [128, 512, 128]
embedding_hidden_dim_4q = [1024, 512, 128]; hidden_layers_4q = [128, 512, 1024, 512, 128]

console = Console()

if __name__ == "__main__":
    if len(sys.argv) != 6: 
        print("Usage: python main.py <number_of_qubits> <n_systems> <n_copies> <n_shots> <model>")
        print("Example: python script.py 100 50 100")
        print("Or:      python script.py 100,200 50,75 100,150")
        sys.exit(1)

    n = int(sys.argv[1])
    n_systems = int(sys.argv[2])
    n_copies = int(sys.argv[3])
    n_shots = int(sys.argv[4])
    model_type = sys.argv[5]
    
    if model_type == "MLP":
        output_base = f"models/outputs/{n}q/outputs_MLP"
    elif model_type == "stella":
        output_base = f"models/outputs/{n}q/outputs_attention"

    os.makedirs(output_base, exist_ok=True)

    _, _, _, _, unitaries, probs, renyi = load_or_generate_dataset(n, n_systems, n_copies, n_shots, generateDataset, f"models/outputs/{n}q/training_data")

    if n != 4:
        probs = probs.sum(axis=2) / n_shots

    dataset = SystemDataset(unitaries, probs, renyi)
    
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.3 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    rng = jax.random.PRNGKey(0)
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs available.")
        device = jax.devices("gpu")[0]
    else:
        print(f"Using CPU.")
        device = jax.devices("cpu")[0]

    unitary_dim = 4**n; outcome_dim = 2**n

    model = None

    if n == 2:
        embedding_hidden_dim = embedding_hidden_dim_2q
        hidden_layers = hidden_layers_2q
    elif n == 4:
        embedding_hidden_dim = embedding_hidden_dim_4q
        hidden_layers = hidden_layers_4q

    if model_type == "MLP":
        model = UnitaryMLP(unitary_dim=unitary_dim, outcome_dim=outcome_dim, embedding_hidden_dim=embedding_hidden_dim, hidden_layers=hidden_layers, output_dim=1)
        state = create_state_UnitaryMLP(rng, model, learning_rate_MLP, (batch_size, n_copies, unitary_dim), (batch_size, n_copies, outcome_dim), num_epochs, device=device)

    elif model_type == "stella": 
        model = UnitaryAttention(unitary_dim=unitary_dim, outcome_dim=outcome_dim, embedding_hidden_dim=embedding_hidden_dim, hidden_layers=hidden_layers, output_dim=1)
        state = create_state_attention(rng, model, learning_rate=learning_rate_stella, 
                                    unitary_shape=(batch_size, n_copies, unitary_dim), 
                                    outcome_shape=(batch_size, n_copies, outcome_dim), 
                                    device = device)
    
    del rng

    print("Parameter Count: ", sum(x.size for x in jax.tree_util.tree_leaves(state.params)))

    start_time = time.time()

    metrics_history = {'train_loss': [], 'eval_loss': []}

    best_loss = jnp.inf; patience_counter = 0; PATIENCE_EPOCHS = 10000

    for epoch in track(range(num_epochs), description="Training model..."):
        train_losses = []
        for unitary_batch, outcomes_batch, renyi_batch in train_loader:
            this_batch = {
                "unitaries": jax.device_put(jnp.asarray(unitary_batch), device),
                "outcomes": jax.device_put(jnp.asarray(outcomes_batch), device),
                "renyi": jax.device_put(jnp.asarray(renyi_batch), device)
            }

            if model_type == "MLP":
                state = train_step_UnitaryMLP(state, this_batch)
                train_metrics = eval_step_UnitaryMLP(state, this_batch)

            elif model_type == "stella":
                state = train_step_attention(state, this_batch)
                train_metrics = eval_step_attention(state, this_batch)

            train_losses.append(train_metrics["loss"])
        
        train_loss = sum(train_losses)/len(train_losses)

        if epoch % eval_after == 0:
            eval_losses = []
            for eval_unitary_batch, eval_outcomes_batch, eval_renyi_batch in eval_loader:
                eval_unitary_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), eval_unitary_batch)
                eval_outcomes_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), eval_outcomes_batch)
                eval_renyi_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), eval_renyi_batch)

                eval_batch = {
                    "unitaries": jnp.asarray(eval_unitary_batch_jax),
                    "outcomes": jnp.asarray(eval_outcomes_batch_jax),
                    "renyi": jnp.asarray(eval_renyi_batch_jax)
                }
                if model_type == "MLP":
                    eval_metric = eval_step_UnitaryMLP(state, eval_batch)
                elif model_type == "stella":
                    eval_metric = eval_step_attention(state, eval_batch)
                eval_losses.append(eval_metric["loss"])

            eval_loss = sum(eval_losses)/len(eval_losses)

            # Logging
            with jax.disable_jit():
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}")
                metrics_history["train_loss"].append(float(train_loss))
                metrics_history["eval_loss"].append(float(eval_loss))

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = state
        
        else:
            patience_counter += 1

        if patience_counter > PATIENCE_EPOCHS:
            break

    training_end = time.time()

    np.savetxt(f"{output_base}/training_loss", metrics_history["train_loss"])
    np.savetxt(f"{output_base}/eval_loss", metrics_history["eval_loss"])

    #Testing
    console.rule(f"[bold blue]Evaluating over unitaries and shots")

    loss_stats = []

    _, _, _, _, test_unitaries, test_probs, test_renyi = load_or_generate_dataset(n, 1000, n_copies, n_shots, generateDataset, f"models/outputs/{n}q/testing_data")

    for shot in np.logspace(np.log10(1), stop=np.log10(n_shots), num = 10, dtype = int):
        losses = []
        angles = []

        for u in np.linspace(1, stop=n_copies, num = 10, dtype = int):
            
            this_test_unitaries = test_unitaries[:, :u, :]
            this_test_probs = test_probs[:, :u, :shot, :]
            this_test_probs = this_test_probs.sum(axis = 2)/shot

            for repeat in range(repeats_per_shot):
                test_dataset = SystemDataset(this_test_unitaries, this_test_probs, test_renyi)
                test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

                test_loss, predicted_vals, target_vals = test_model(best_state, test_dataloader, model_type)
                losses.append(test_loss)

                pred_flat = np.concatenate([x.ravel() for x in predicted_vals])
                true_flat = np.concatenate([x.ravel() for x in target_vals])

                angles.append(analyze_prediction_vs_true(pred_flat, true_flat))

                if repeat == 0:  # Save predictions only once per shot
                    np.savetxt(f"{output_base}/{shot}shot_{u}unitary_predicted.txt", pred_flat)
                    np.savetxt(f"{output_base}/{shot}shot_{u}unitary_true.txt", true_flat)
        
            # Mean/std over repeats for this shot count
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            angle = np.mean(angles)
            angle_std = np.std(angles)
            loss_stats.append([shot, u, mean_loss, std_loss, angle, angle_std])

            console.log(
                f"[green]Shots:[/green] {shot:<5} | "
                f"[cyan]Copies:[/cyan] {u:<2} | "
                f"[yellow]Loss:[/yellow] {mean_loss:.6f} ± {std_loss:.6f} | "
                f"[blue]Angle: [/blue] {angle:.6f} ± {angle_std:.6f}"
                )
            

    # Save full results
    loss_stats = np.array(loss_stats)
    np.savetxt(f"{output_base}/testing_loss.txt", loss_stats)

    print(f"Training completed in {training_end - start_time:.2f} seconds.")

    if model_type == "stella":
        np.savetxt(f"{output_base}/attention_matrix", state.mutable_state)


