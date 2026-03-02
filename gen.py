import numpy as np
from minst_test import x as X_large
import matplotlib.pyplot as plt
def relu(x):
    """ReLu activation, it keeps postive values (designated by the 0) and zeroes out negatives"""
    return np.maximum(0,x) 
def relu_der(x):
    """Derivative for backprop"""
    return (x>0).astype(float)

def reparm(mean,logvar):
    """
    INput: mean, center of distrubuiton
    logvar: shape-log of variance

    Output:
    z:shape - sampled latent vector
    """

    #convert log-variance to standard devation

    std = np.exp(0.5 *logvar)
    epsilon = np.random.randn(*mean.shape)
    #formula now z ~ mean,variance) but we can backprop through mean and std
    z = mean + std * epsilon

    return z

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def reparameterize(mean, logvar):
    """The reparameterization trick"""
    std = np.exp(0.5 * logvar)
    epsilon = np.random.randn(*mean.shape)
    z = mean + std * epsilon
    return z

def sigmoid_derivative(x):
    return x * (1 - x)

class Encoder:
    def __init__(self, input_size=784, hidden_size=256, latent_size=28):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # First layer: 784 -> 256
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1./input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Mean path: 256 -> 28
        self.w_mean = np.random.randn(hidden_size, latent_size) * np.sqrt(1./hidden_size)
        self.b_mean = np.zeros((1, latent_size))
        
        # Log-variance path: 256 -> 28
        self.w_logvar = np.random.randn(hidden_size, latent_size) * np.sqrt(1./hidden_size)
        self.b_logvar = np.zeros((1, latent_size))
    
    def foward(self, x):
        self.x = x
        self.h1 = np.dot(x, self.w1) + self.b1
        self.h1_activated = relu(self.h1)
        
        self.z_mean = np.dot(self.h1_activated, self.w_mean) + self.b_mean
        self.z_logvar = np.dot(self.h1_activated, self.w_logvar) + self.b_logvar
        
        return self.z_mean, self.z_logvar




class Decoder:
    def __init__(self, latent_size=28, hidden_size=256, output_size=784):
        """
        input_size: 784 pixels (which is 28x28)
        hidden size: 256 (which is a power of 2 and the intermediate layer size)
        latent_size: how many dimmensions in out compressed space
        """
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        #we begin with our first layer, compression from 784 to 256
        self.w1 = np.random.randn(latent_size, hidden_size) * np.sqrt(1./latent_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Second layer: expand from 256 -> 784
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1./hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        #this is the second layer, it splits into two paths
        #path A: mean of latent distriibution (256-> 28)
        self.w_mean = np.random.randn(hidden_size,latent_size) *np.sqrt(1./hidden_size)
        self.b_mean = np.zeros((1,latent_size))
        
        #path B: log varations of latent distribution from 256 to 28
        self.w_logvar = np.random.randn(hidden_size,latent_size)*np.sqrt(1./hidden_size)
        self.b_logvar = np.zeros((1,latent_size))
    
    def foward(self,z):
        """Input: x is a bath of images, 
        output: mean and log-variance
        """
        #save input for backprop later
        self.z = z

        #hidden layer with ReLU activation
        self.h1 = np.dot(z,self.w1)+self.b1
        self.h1_activated = relu(self.h1)

        self.output = np.dot(self.h1_activated, self.w2) + self.b2
        self.reconstructed = sigmoid(self.output)

        return self.reconstructed



class VAE:
    def __init__(self, input_size = 784, hidden_size = 256, latent_size = 28):
        self.encoder = Encoder(input_size, hidden_size,latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
    def foward(self,x):
        """full foward pass through the VAE"""
        #step 1, encode the get mean and logvar
        self.z_mean, self.z_logvar = self.encoder.foward(x)
        #step 2, sample from latent space using reparm trick
        self.z = reparameterize(self.z_mean,self.z_logvar)
        #step 3: decode back to image
        self.reconstructed = self.decoder.foward(self.z)

        return self.reconstructed    
    def compute_loss(self,x):
        """
        VAE has two components:
        1.Reconstuction loss- how well can we rebuild the original image?
        
        2. KL Divergence Loss- keeps latent space organized and smooth measures how diffrent out latent distribution is from N(0,1)"""

        batch_size = x.shape[0]


        #loss 1: recontuction loss
        recon_loss = -np.sum(
            x*np.log(self.reconstructed + 1e-8)+
            (1-x)*np.log(1-self.reconstructed +1e-8)
        ) / batch_size


        #loss 2: KL Divergence loss

        kl_loss = -0.5 *np.sum(1+self.z_logvar-self.z_mean**2-np.exp(self.z_logvar))/batch_size

        total_loss = recon_loss +kl_loss

        return total_loss, recon_loss, kl_loss
    def backward(self, x, learning_rate):
        """Backpropagation through the VAE - the nudgeing man!"""
        batch_size = x.shape[0]
        
        # ========== DECODER GRADIENTS ==========
        # Gradient of reconstruction loss w.r.t. reconstructed output
        d_reconstructed = -(x / (self.reconstructed + 1e-8) - 
                           (1 - x) / (1 - self.reconstructed + 1e-8)) / batch_size
        d_reconstructed = d_reconstructed * sigmoid_derivative(self.reconstructed)
        
        # Decoder layer 2 (hidden -> output)
        grad_w2 = np.dot(self.decoder.h1_activated.T, d_reconstructed)
        grad_b2 = np.sum(d_reconstructed, axis=0, keepdims=True)
        
        # Decoder layer 1 (latent -> hidden)
        d_dec_h1 = np.dot(d_reconstructed, self.decoder.w2.T)
        d_dec_h1 = d_dec_h1 * relu_der(self.decoder.h1_activated)
        
        grad_w1_dec = np.dot(self.z.T, d_dec_h1)
        grad_b1_dec = np.sum(d_dec_h1, axis=0, keepdims=True)
        
        # Gradient flowing back to z
        d_z_recon = np.dot(d_dec_h1, self.decoder.w1.T)
        
        # ========== KL DIVERGENCE GRADIENTS ==========
        # Gradient of KL loss w.r.t. z, z_mean, z_logvar
        d_z_mean_kl = self.z_mean / batch_size
        d_z_logvar_kl = 0.5 * (np.exp(self.z_logvar) - 1) / batch_size
        
        # Reparameterization trick gradients
        std = np.exp(0.5 * self.z_logvar)
        epsilon = (self.z - self.z_mean) / (std + 1e-8)
        
        d_z_mean_reparam = d_z_recon
        d_z_logvar_reparam = d_z_recon * epsilon * 0.5 * std
        
        # Combine gradients
        d_z_mean = d_z_mean_reparam + d_z_mean_kl
        d_z_logvar = d_z_logvar_reparam + d_z_logvar_kl
        
        # ========== ENCODER GRADIENTS ==========
        # Encoder mean path
        d_enc_h1_mean = np.dot(d_z_mean, self.encoder.w_mean.T)
        grad_w_mean = np.dot(self.encoder.h1_activated.T, d_z_mean)
        grad_b_mean = np.sum(d_z_mean, axis=0, keepdims=True)
        
        # Encoder logvar path
        d_enc_h1_logvar = np.dot(d_z_logvar, self.encoder.w_logvar.T)
        grad_w_logvar = np.dot(self.encoder.h1_activated.T, d_z_logvar)
        grad_b_logvar = np.sum(d_z_logvar, axis=0, keepdims=True)
        
        # Encoder hidden layer
        d_enc_h1 = (d_enc_h1_mean + d_enc_h1_logvar) * relu_der(self.encoder.h1_activated)
        grad_w1_enc = np.dot(x.T, d_enc_h1)
        grad_b1_enc = np.sum(d_enc_h1, axis=0, keepdims=True)
        
        # ========== UPDATE WEIGHTS (the nudgeing!) ==========
        # Decoder updates
        self.decoder.w2 -= learning_rate * grad_w2
        self.decoder.b2 -= learning_rate * grad_b2
        self.decoder.w1 -= learning_rate * grad_w1_dec
        self.decoder.b1 -= learning_rate * grad_b1_dec
        
        # Encoder updates
        self.encoder.w_mean -= learning_rate * grad_w_mean
        self.encoder.b_mean -= learning_rate * grad_b_mean
        self.encoder.w_logvar -= learning_rate * grad_w_logvar
        self.encoder.b_logvar -= learning_rate * grad_b_logvar
        self.encoder.w1 -= learning_rate * grad_w1_enc
        self.encoder.b1 -= learning_rate * grad_b1_enc

if __name__ == "__main__":
    print("="*60)
    print("TESTING THE COMPLETE VAE")
    print("="*60)
    
    input_size = 784
    hidden_size = 256
    latent_size = 28
    learning_rate = 0.0001
    epochs = 20
    batch_size = 32

    # Create VAE
    vae = VAE(input_size=784, hidden_size=256, latent_size=28)
    
    for e in range(epochs):
        total_loss_epoch = 0
        recon_loss_epoch = 0
        kl_loss_epoch = 0
        num_batches = 0
        
        # Loop through batches
        for i in range(0, 60000, batch_size):  # Using first 60k for training
            x_batch = X_large[i:i+batch_size]
            
            # Forward pass
            reconstructed = vae.foward(x_batch)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae.compute_loss(x_batch)
            
            # Backward pass (the nudgeing!)
            vae.backward(x_batch, learning_rate)
            
            # Track losses
            total_loss_epoch += total_loss
            recon_loss_epoch += recon_loss
            kl_loss_epoch += kl_loss
            num_batches += 1
        
        # Average losses for the epoch
        avg_total = total_loss_epoch / num_batches
        avg_recon = recon_loss_epoch / num_batches
        avg_kl = kl_loss_epoch / num_batches
        
        print(f"Epoch {e+1}/{epochs} - Total Loss: {avg_total:.2f}, Recon: {avg_recon:.2f}, KL: {avg_kl:.2f}")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print("Beagle has learned to generate numbers!")
    print("\nSaving the brain...")

    np.save('vae_encoder_w1.npy', vae.encoder.w1)
    np.save('vae_encoder_b1.npy', vae.encoder.b1)
    np.save('vae_encoder_w_mean.npy', vae.encoder.w_mean)
    np.save('vae_encoder_b_mean.npy', vae.encoder.b_mean)
    np.save('vae_encoder_w_logvar.npy', vae.encoder.w_logvar)
    np.save('vae_encoder_b_logvar.npy', vae.encoder.b_logvar)
    
    np.save('vae_decoder_w1.npy', vae.decoder.w1)
    np.save('vae_decoder_b1.npy', vae.decoder.b1)
    np.save('vae_decoder_w2.npy', vae.decoder.w2)
    np.save('vae_decoder_b2.npy', vae.decoder.b2)
    
    print("Brain saved!")
    
    # Generate some new digits!
    print("\nChecking latent space range...")
    sample_batch = X_large[0:100]
    z_mean_sample, z_logvar_sample = vae.encoder.foward(sample_batch)
    print(f"Mean of z_mean: {np.mean(z_mean_sample):.3f}")
    print(f"Std of z_mean: {np.std(z_mean_sample):.3f}")
    print("\nGenerating Beagle's favorite numbers...")
    for i in range(5):
        # Sample random point from latent space N(0,1)
        random_z = np.random.randn(1, latent_size)*0.5
        generated = vae.decoder.foward(random_z)
        generated_image = generated.reshape(28, 28)
        
        print(f"\nGenerated digit #{i+1}:")
        for row in generated_image:
            line = ""
            for pixel in row:
                if pixel < 0.3:
                    line += " "
                elif pixel < 0.5:
                    line += "."
                elif pixel < 0.7:
                    line += "o"
                else:
                    line += "#"
            print(line)
