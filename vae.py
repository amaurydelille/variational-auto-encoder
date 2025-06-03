import numpy as np
import keras
from keras.datasets import mnist
import os

EPOCHS = 10
BATCH_SIZE = 128

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    t = np.tanh(z)
    return 1 - t ** 2

def binary_cross_entropy(x, x_reconstructed):
    x_reconstructed = np.clip(x_reconstructed, 1e-7, 1 - 1e-7)
    return - np.sum(x * np.log(x_reconstructed) + (1 - x) * np.log(1 - x_reconstructed))

def kl_divergence(mu, logvar):
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))

class GaussianEncoder:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lr=1e-4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, ))
        self.b_mu = np.zeros((latent_dim, ))
        self.b_logvar = np.zeros((latent_dim, ))

    def inference(self, x: np.ndarray):
        self.x = x
        self.h_pre = self.W1 @ x + self.b1
        self.h = tanh(self.h_pre)
        self.mu = self.W_mu.T @ self.h + self.b_mu
        self.logvar = self.W_logvar.T @ self.h + self.b_logvar
        return self.mu, self.logvar
    
    def sample_z(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon
        self.epsilon = epsilon
        return z

    def backpropagation(self, x, x_reconstructed, mu, logvar, z, decoder, lr=1e-4):
        d_recon = (x_reconstructed - x) / (x_reconstructed * (1 - x_reconstructed) + 1e-8)
        dz = decoder.W1 @ (tanh_derivative(decoder.x_reconstructed_pre) * d_recon)
        dmu = (mu - 0) + dz
        dlogvar = 0.5 * (np.exp(logvar) - 1) + 0.5 * dz * self.epsilon * np.exp(0.5 * logvar)
        dh = self.W_mu @ dmu + self.W_logvar @ dlogvar
        dh_pre = dh * tanh_derivative(self.h_pre)
        self.W_mu -= lr * np.outer(self.h, dmu)
        self.b_mu -= lr * dmu
        self.W_logvar -= lr * np.outer(self.h, dlogvar)
        self.b_logvar -= lr * dlogvar
        self.W1 -= lr * np.outer(dh_pre, x)
        self.b1 -= lr * dh_pre

class GaussianDecoder:
    def __init__(self, latent_dim: int, output_dim: int, lr=1e-4):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lr = lr
        self.W1 = np.random.randn(latent_dim, output_dim) * 0.01
        self.b1 = np.random.randn(output_dim) * 0.01
    
    def inference(self, z):
        self.z = z
        self.x_reconstructed_pre = self.W1.T @ z + self.b1
        self.x_reconstructed = sigmoid(self.x_reconstructed_pre)
        return self.x_reconstructed

    def compute_loss(self, x, x_reconstructed, mu, logvar):
        reconstruction_loss = binary_cross_entropy(x, x_reconstructed)
        kl_loss = kl_divergence(mu, logvar)
        return reconstruction_loss + kl_loss
    
    def backpropagation(self, x, x_reconstructed, mu, logvar, z, lr=1e-4):
        d_recon = (x_reconstructed - x) / (x_reconstructed * (1 - x_reconstructed) + 1e-8)
        dx_reconstructed_pre = d_recon * (x_reconstructed * (1 - x_reconstructed))
        self.W1 -= lr * np.outer(self.z, dx_reconstructed_pre)
        self.b1 -= lr * dx_reconstructed_pre

class VAE:
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, lr=1e-4) -> None:
        self.encoder = GaussianEncoder(input_dim, hidden_dim, latent_dim, lr)
        self.decoder = GaussianDecoder(latent_dim, input_dim, lr)
        self.lr = lr
    
    def forward(self, x: np.ndarray):
        mu, logvar = self.encoder.inference(x)
        z = self.encoder.sample_z(mu, logvar)
        x_reconstructed = self.decoder.inference(z)
        return x_reconstructed, mu, logvar, z
    
    def compute_loss(self, x: np.ndarray, x_reconstructed, mu, logvar):
        return self.decoder.compute_loss(x, x_reconstructed, mu, logvar)
    
    def backpropagation(self, x: np.ndarray):
        x_reconstructed, mu, logvar, z = self.forward(x)
        loss = self.compute_loss(x, x_reconstructed, mu, logvar)
        self.decoder.backpropagation(x, x_reconstructed, mu, logvar, z, self.lr)
        self.encoder.backpropagation(x, x_reconstructed, mu, logvar, z, self.decoder, self.lr)
        return loss
    
    def train(self, X: np.ndarray, epochs: int):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                x = X[i]
                loss = self.backpropagation(x)
                total_loss += loss
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    
    def test(self, X: np.ndarray):
        for i in range(10):
            x = X[i]
            x_reconstructed, _, _, _ = self.forward(x)
            import matplotlib.pyplot as plt
            plt.imshow(x.reshape(28, 28), cmap='gray')
            plt.show()
            plt.imshow(x_reconstructed.reshape(28, 28), cmap='gray')
            plt.show()
    
    def save_weights(self, filepath: str):
        np.savez(filepath,
                 encoder_W1=self.encoder.W1,
                 encoder_W_mu=self.encoder.W_mu,
                 encoder_W_logvar=self.encoder.W_logvar,
                 encoder_b1=self.encoder.b1,
                 encoder_b_mu=self.encoder.b_mu,
                 encoder_b_logvar=self.encoder.b_logvar,
                 decoder_W1=self.decoder.W1,
                 decoder_b1=self.decoder.b1)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        data = np.load(filepath)
        self.encoder.W1 = data['encoder_W1']
        self.encoder.W_mu = data['encoder_W_mu']
        self.encoder.W_logvar = data['encoder_W_logvar']
        self.encoder.b1 = data['encoder_b1']
        self.encoder.b_mu = data['encoder_b_mu']
        self.encoder.b_logvar = data['encoder_b_logvar']
        self.decoder.W1 = data['decoder_W1']
        self.decoder.b1 = data['decoder_b1']
        print(f"Weights loaded from {filepath}")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    vae = VAE(784, 30, 10)
    
    if os.path.exists("/home/amaury-delille/Documents/machine_learning/ml-dl/vae/vae_weights.npz"):
        vae.load_weights("/home/amaury-delille/Documents/machine_learning/ml-dl/vae/vae_weights.npz")
    else:
        vae.train(x_train, EPOCHS)
        vae.save_weights("/home/amaury-delille/Documents/machine_learning/ml-dl/vae/vae_weights.npz")
    
    x_reconstructed, mu, logvar, z = vae.test(x_test)
    