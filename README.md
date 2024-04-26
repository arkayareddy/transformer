Transformers for Next Character Prediction

Flax is a high-performance neural network library and ecosystem for JAX that is designed for flexibility.

JAX is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

Parameters used:
# Random seed
SEED = 42

# Learning rate passed to the optimizer
LEARNING_RATE = 0.005

# Batch size
BATCH_SIZE = 128  # Note: This value is redefined from 128 to 512

# Number of training iterations
N_ITERATIONS = 6000

# Number of training iterations between two consecutive evaluations
N_FREQ_EVAL = 200

# Rate for dropout in the transformer model
DROPOUT_RATE = 0.2

# Context window for the transformer model
BLOCK_SIZE = 64

# Number of layers for the transformer model
NUM_LAYERS = 6

# Number of Epochs
NUM_EPOCHS = 10

# Size of the embedding for the transformer model
EMBED_SIZE = 256

# Number of heads for the transformer model
NUM_HEADS = 8

# Size of the heads for the transformer model
HEAD_SIZE = 32

Length of training and validation text characters
Length of text for training: 1_059_624 characters 
Length of text for validation: 55_770 characters

Vocabulary present in the data 
Vocabulary:,  
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
Length of vocabulary:  65
Total model parameters
Total number of parameters: 4,788,801

Explanation of Components:

Model Initialization: NanoLM is initialized with specific hyperparameters that define its structure. These parameters include aspects that influence the model's capacity (e.g., num_layers, embed_size) and its ability to generalize (e.g., dropout_rate).

Loss Function (loss_fun): This function is crucial for training as it provides a quantitative measure of how well the model's predictions match the expected outputs. The use of dropout during training (training=True) helps prevent overfitting.

Evaluation Step (eval_step): This function computes the loss for evaluation purposes. Since dropout is typically not used during evaluation or testing (to get deterministic results), it is disabled here (training=False).

Just-In-Time Compilation (@jax.jit): The use of jax.jit on eval_step compiles this function into a highly optimized form, speeding up its execution significantly, which is particularly useful during the repeated evaluations over validation data.

Explanation of Components:

PRNG Key Initialization: jax.random.PRNGKey(SEED) creates a pseudo-random number generator key based on a seed value (SEED). This is important for ensuring reproducibility in experiments as it controls the randomness in model initialization.

Key Splitting: jax.random.split(key) is used to generate new subkeys from an original key. In JAX, every random operation requires a fresh key, and splitting helps in managing these keys without reusing them, thus avoiding unwanted correlations in random operations.

Model Parameter Initialization: model.init() initializes the model's parameters. This step typically involves setting up weights and biases using random values generated based on the key. The input jnp.ones((BATCH_SIZE, BLOCK_SIZE), dtype=jnp.int32) simulates an input batch to define how large the inputs to the model will be, which is crucial for defining the shapes of parameters like weights in neural networks.
Losses:

Step: 0 train loss: 4.587119102478027 eval loss: 6.042149543762207 
Step: 2000 train loss: 1.388554573059082 eval loss: 1.4247114658355713 
Step: 4000 train loss: 1.2833781242370605 eval loss: 1.3967227935791016 

CPU times: user 4min 16s, sys: 3min 41s, total: 7min 57s Wall time: 7min 52s

Sample output generated:

CAMILLO: 
Yes, sir, bestrictle. 

LEONTES: 
For which I can scoffer. 

Clown: 
I shall be full of bosoms in this grief, And throw the droping sords give thee this banish'd: The murderer whereof he's slain are so At gates; the dread ham, on catched of wrongs: That I trew my words from heaven's father, His hug-lips in so face, if his world, Made it kindly know what he most Covenance on their lips I for link not To our great speech. 

Messenger: 
My gracious liege, and then I love thee sir; Come about the conspiracy. Thus thus they are the vice music ere thou shalt not abuse thy head? Be thy departing, but at night winter peace, Should be welcome; thy life, thy trembling peace, Comes, give me down as the new-day. 

Provost: 
Pardon me, cousin! 

DUKE VINCENTIO: 
The princes shall be gone. 

Provost: 
I have rather since, the drawbring state of eldest faces is but within such same long. 

OXTON: 
Arise, for that yet did follow all, Since yet over--Pluck the action! O, not some other the present. Where Me

Transformers for Translation

French-English translation

Train and Validation Losses

Epoch: 1, Train loss: 0.810, Val loss: 1.410, Epoch time = 19.292s 
Epoch: 2, Train loss: 0.761, Val loss: 1.421, Epoch time = 19.433s 
Epoch: 3, Train loss: 0.715, Val loss: 1.397, Epoch time = 19.517s 
Epoch: 4, Train loss: 0.674, Val loss: 1.412, Epoch time = 19.564s 
Epoch: 5, Train loss: 0.635, Val loss: 1.404, Epoch time = 19.583s 
Epoch: 6, Train loss: 0.598, Val loss: 1.412, Epoch time = 19.590s 
Epoch: 7, Train loss: 0.563, Val loss: 1.415, Epoch time = 19.736s 
Epoch: 8, Train loss: 0.530, Val loss: 1.415, Epoch time = 19.652s 
Epoch: 9, Train loss: 0.499, Val loss: 1.431, Epoch time = 19.812s 
Epoch: 10, Train loss: 0.468, Val loss: 1.442, Epoch time = 19.638s

Sample Output:

French sentence: “Un groupe de personnes se tient devant un igloo .”
English sentence: “A group of people stand in front of an igloo .”

French sentence: “Il chante dans la chorale .”
English sentence: “He is singing in the choir . ”

English-French Translation

Train and Validation Losses

Epoch: 1, Train loss: 3.249, Val loss: 2.723, Epoch time = 20.590s 
Epoch: 2, Train loss: 2.522, Val loss: 2.206, Epoch time = 20.501s 
Epoch: 3, Train loss: 2.081, Val loss: 1.908, Epoch time = 20.369s 
Epoch: 4, Train loss: 1.783, Val loss: 1.713, Epoch time = 20.305s 
Epoch: 5, Train loss: 1.561, Val loss: 1.576, Epoch time = 20.377s 
Epoch: 6, Train loss: 1.391, Val loss: 1.479, Epoch time = 20.530s 
Epoch: 7, Train loss: 1.254, Val loss: 1.413, Epoch time = 20.515s 
Epoch: 8, Train loss: 1.140, Val loss: 1.357, Epoch time = 20.435s 
Epoch: 9, Train loss: 1.045, Val loss: 1.313, Epoch time = 20.418s 
Epoch: 10, Train loss: 0.964, Val loss: 1.263, Epoch time = 20.337s
Sample Output:

English sentence: “A group of people talking.”
French sentence: “Un groupe de personnes parlant .”

English sentence: “He sings in the choir.”
French sentence: “Il chante dans la chorale .”
![image](https://github.com/arkayareddy/transformer/assets/52198120/8b60f2ac-67bc-4eb9-9429-334c34d4e63e)
