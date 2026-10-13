import numpy as np

print("--- Simple Attention Mechanism ---")

# Encoder output vectors for four input words.
encoder_outputs = np.array([
    [0.1, 0.2],   # I
    [0.5, 0.3],   # am
    [0.2, 0.8],   # a
    [0.9, 0.1]    # student
])

print("\nEncoder outputs (4 words, 2 features each):")
print(encoder_outputs)

# Decoder hidden state: what the decoder is currently looking for.
decoder_hidden_state = np.array([0.7, 0.4])
print("\nDecoder hidden state:")
print(decoder_hidden_state)

# Dot product gives a simple alignment/similarity score.
alignment_scores = np.dot(encoder_outputs, decoder_hidden_state)
print("\nAlignment scores:")
print(alignment_scores)

# Convert scores to probabilities.
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # numerical stability
    return exp_x / exp_x.sum()

attention_weights = softmax(alignment_scores)
print("\nAttention weights:")
print(attention_weights)
print("Sum of attention weights:", attention_weights.sum())

# Weighted sum of encoder outputs.
context_vector = np.sum(
    encoder_outputs * attention_weights[:, np.newaxis],
    axis=0
)

print("\nContext vector:")
print(context_vector)

# Show which input position received the highest attention.
words = ["I", "am", "a", "student"]
most_attended = words[int(np.argmax(attention_weights))]
print("\nMost attended word:", most_attended)

print("\n--- How attention works ---")
print("1. The encoder produces a vector for each input word.")
print("2. The decoder has a hidden state representing its current need.")
print("3. Alignment scores compare each encoder vector with the decoder state.")
print("4. Softmax converts the scores into attention weights.")
print("5. A weighted sum produces the context vector.")
