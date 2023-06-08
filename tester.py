import torch
import torch.nn as nn
from attention import ScaledDotProductAttention


def main(): 
    # Create a sample input
    batch_size, seq_length, hidden_dim = 2, 3, 4
    query = torch.randn(batch_size, seq_length, hidden_dim)
    key = torch.randn(batch_size, seq_length, hidden_dim)
    value = torch.randn(batch_size, seq_length, hidden_dim)

    # Create the ScaledDotProductAttention module
    attention = ScaledDotProductAttention()
    # attention = torch.nn.functional.scaled_dot_product_attention

    # Run the forward pass
    output, attention_weights = attention(query, key, value)

    # Print the output and attention weights
    print("Output:", output)
    print("Attention Weights:", attention_weights)

    # Check the output shapes
    print("Output Shape:", output.shape)
    print("Attention Weights Shape:", attention_weights.shape)

    # Check that the attention weights sum to 1 along the last dimension
    attention_weights_sum = attention_weights.sum(dim=-1)
    print("Attention Weights Sum:", attention_weights_sum)
    print("All attention weights sum to 1:", torch.allclose(attention_weights_sum, torch.ones_like(attention_weights_sum)))


if __name__ == "__main__":
    main()