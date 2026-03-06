import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence

class MLP(nn.Module):
    input_dim: int
    hidden_layer_dim: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for layer in self.hidden_layer_dim:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)        
        x = nn.Dense(self.output_dim)(x)
    
        return x
    
class ResidualBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        if residual.shape[-1] != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim)(residual)

        return x + residual


def process_one(unitary_slice, outcome_slice, unitary_dim, embedding_hidden_dim, outcome_dim, hidden_layers, output_dim):
    """
    Processes one (unitary_slice, outcome_slice) pair.
    Keeps hidden_dim constant so residual connections work.
    """
    # Embed the unitary
    embed_unitary = MLP(unitary_dim, embedding_hidden_dim, outcome_dim)(unitary_slice)

    # Concatenate
    x = jnp.concatenate([embed_unitary, outcome_slice], axis=-1)

    for dim in hidden_layers:
        x = ResidualBlock(dim)(x)

    # Output layer
    x = nn.Dense(output_dim)(x)
    return x
 

class UnitaryMLP(nn.Module):
    unitary_dim: int
    outcome_dim: int
    embedding_hidden_dim: Sequence[int]
    hidden_layers: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, unitary, outcome, compute_attn: bool = True):
        mapped_fn = jax.vmap(
            lambda u, o: process_one(u, o, self.unitary_dim, self.embedding_hidden_dim, self.outcome_dim, self.hidden_layers, self.output_dim),
            in_axes=(1, 1), out_axes=1)

        preds = mapped_fn(unitary, outcome)  # [batch, sequence, output_dim]
        preds_mean = jnp.mean(preds, axis=1)
        return preds_mean

def scaled_dot_product(q, k):
    attn_matrix = jnp.einsum('bmn, bpq -> bnp', q, jnp.swapaxes(k, 1, 2))
    attn_matrix /= jnp.sqrt(q.shape[-1])

    return nn.softmax(attn_matrix)


class UnitaryAttention(nn.Module):
    unitary_dim: int
    outcome_dim:int
    hidden_layers: Sequence[int]
    embedding_hidden_dim: Sequence[int]
    output_dim: int

    def setup(self):
        self.UnitaryEmbedding = MLP(self.unitary_dim, self.embedding_hidden_dim, self.outcome_dim)
        
    @nn.compact
    def __call__(self, unitary: jax.Array, probability: jax.Array, compute_attn: bool = True):

        attn_matrix_var = self.variable('state', 'Attention Matrix', nn.initializers.glorot_uniform(), key = jax.random.key(42), shape = (self.outcome_dim, self.outcome_dim), dtype = jnp.complex64)

        embedded_unitary = self.UnitaryEmbedding(unitary)

        #Self attention over the unitaries and compute the embedding space during training
        if compute_attn:
            attn_matrix = scaled_dot_product(embedded_unitary, embedded_unitary)
            attn_matrix_var.value = attn_matrix[0]

        attn_matrix_batched = jnp.broadcast_to(attn_matrix_var.value, (unitary.shape[0], *attn_matrix_var.value.shape))

        relation_vec = jnp.einsum('bcd, bnp -> bcp', embedded_unitary, attn_matrix_batched)
        cat_input = jnp.concatenate([relation_vec, probability], axis = -1)
    
        def mlp_forward(x):
            for layer in self.hidden_layers:
                residual = x
                x = nn.Dense(layer)(x)
                x = nn.relu(x)
                x = nn.LayerNorm()(x)

                # Project residual if needed
                if residual.shape[-1] != x.shape[-1]:
                    residual = nn.Dense(x.shape[-1])(residual)

                x = x + residual
            x = nn.Dense(self.output_dim)(x)
            return x

        mlp_vmap = jax.vmap(mlp_forward, in_axes=0, out_axes=0)
        preds = mlp_vmap(cat_input)

        return jnp.mean(preds, axis=1)