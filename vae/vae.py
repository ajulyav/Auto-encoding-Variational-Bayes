# Implements auto-encoding variational Bayes.

import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
    
from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images

def save_images(images, file_name): return s_images(images, file_name, vmin = 0.0, vmax = 1.0)

# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)

# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale = 1e-2):

    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),   # weight matrix
             scale * npr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)         # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs

# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):

    # Params of a diagonal Gaussian.

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    batch_size, latent_dim = mean.shape[0], mean.shape[1]
    samples = npr.randn(batch_size, latent_dim)

    return samples * np.exp(log_std) + mean

# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):

    label_prob = targets*sigmoid(logits)+(1-targets)*(1-sigmoid(logits))

    label_prob = np.log(label_prob)

    return np.sum(label_prob, axis=-1)

# This evaluates the KL between q and the prior

def compute_KL(q_means_and_log_stds):
    
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]
    
    std = np.exp(log_std)

    KL = np.sum(0.5*(np.square(std) + (np.square(mean) - 1 - 2*log_std)),axis=-1)

    #KL /= 32 * 784 #batch_size*28*28

    return KL

# This evaluates the lower bound

def vae_lower_bound(gen_params, rec_params, data):

    # TODO compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    # 2 - sample the latent variables associated to the batch in data 
    #     (use sample_latent_variables_from_posterior and the encoder output)
    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    # 5 - return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term

    outputs = neural_net_predict(rec_params, data)
  
    latents =  sample_latent_variables_from_posterior(outputs)
    
    preds  = neural_net_predict(gen_params, latents)

    likehood = bernoulli_log_prob(data, preds)

    KL = compute_KL(outputs)

    difference = np.mean(likehood - KL)
    #print(difference)

    return difference

if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0) # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [ latent_dim ] + [ n_units for i in range(n_layers) ] + [ data_dim ]
    rec_layer_sizes = [ data_dim ]  + [ n_units for i in range(n_layers) ] + [ latent_dim * 2 ]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)

    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)

    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params) 

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)

    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params

    def objective(flattened_combined_params):

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data
    
        on = train_images[ data_idx ,: ] > npr.uniform(size = train_images[ data_idx ,: ].shape)
        images = train_images[ data_idx, : ] * 0.0
        images[ on ] = 1.0

        return vae_lower_bound(gen_params, rec_params, images) 

    # Get gradients of objective using autograd.

    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    
    t = 1

    # TODO write here the initial values for the ADAM parameters (including the m and v vectors)
    # you can use np.zeros_like(flattened_current_params) to initialize m and v

    # We do the actual training

    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)
    exp_decay_rate1 = 0.9
    exp_decay_rate2 = 0.999
    epsila  = 0.00000001
    
    for epoch in range(num_epochs):

        elbo_est = 0.0

        for n_batch in range(int(np.ceil(N / batch_size))):

            batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))
            grad = objective_grad(flattened_current_params)

            # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates

            m = exp_decay_rate1*m + (1-exp_decay_rate1)*grad
            v = exp_decay_rate2*v + (1-exp_decay_rate2)*(grad**2)
                       
            m_hat = m/(1-(exp_decay_rate1**t))
            v_hat = v/(1-(exp_decay_rate2**t))
            
            flattened_current_params = flattened_current_params + learning_rate*m_hat/(np.sqrt(v_hat)+epsila)
            elbo_est += objective(flattened_current_params)

            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We obtain the final trained parameters

    gen_params, rec_params = unflat_params(flattened_current_params)

    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images

    latents = npr.randn(25, latent_dim)
    prior_25 = sigmoid(neural_net_predict(gen_params,latents))
    save_images(prior_25, 'prior_25.png')

    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model) 
    # and save them alongside with the original image using save_images

    test_10 = test_images[0:10]
    outputs10 = neural_net_predict(rec_params, test_10)
  
    latents10 =  sample_latent_variables_from_posterior(outputs10)  
    preds10  = sigmoid(neural_net_predict(gen_params, latents10))

    save_images(test_10, 'recognition_10.png')
    save_images(preds10, 'generative_10.png')


    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model) 
    # and save them alongside with the original image using save_images
        
    alpha_values = np.linspace(0, 1, 30)
    D = np.shape(outputs10)[-1] // 2
    latent_mean = outputs10[:, :D]
    vectors = []

    for alpha in alpha_values:

        # Latent space interpolation select latent_mean for image pairs, f.e., 0-1, 2-3, 3-4,...
        vector = latent_mean[0] * (1 - alpha) + latent_mean[1] * alpha
        vectors.append(vector)

        preds_inter  = sigmoid(neural_net_predict(gen_params, vectors))

        #rename the file to save to the different one
        save_images(preds_inter, 'interpolation_0_1.png')

        pass


