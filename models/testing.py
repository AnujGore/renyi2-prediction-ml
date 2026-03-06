import jax.numpy as jnp
from utils import eval_step_UnitaryMLP, eval_step_attention
import jax


def test_model(best_state, test_dataset, model_type):
    
    test_losses = []; target_vals = []; predicted_vals = []

    for test_unitary_batch, test_outcomes_batch, test_entropy_batch in test_dataset:
        test_unitary_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), test_unitary_batch)
        test_outcomes_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), test_outcomes_batch)
        test_entropy_batch_jax = jax.tree_util.tree_map(lambda x: jnp.asarray(x.cpu()), test_entropy_batch)

        test_batch = {
            "unitaries": jnp.asarray(test_unitary_batch_jax),
            "outcomes": jnp.asarray(test_outcomes_batch_jax),
            "renyi": jnp.asarray(test_entropy_batch_jax)
        }

        state = best_state
        if model_type == "MLP":
            test_metric = eval_step_UnitaryMLP(state, test_batch)
        elif model_type == "stella":
            test_metric = eval_step_attention(state, test_batch)
            
        test_loss = test_metric["loss"]
        test_losses.append(test_loss)
        target_vals.append(test_metric["target"])
        predicted_vals.append(test_metric["preds"])

    test_loss = sum(test_losses)/len(test_losses)

    return test_loss, predicted_vals, target_vals