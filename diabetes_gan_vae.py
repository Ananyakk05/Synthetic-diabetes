import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from diabetes import generate_diabetes_data

# Custom dataset class for diabetes data
class DiabetesDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# VAE Model
class DiabetesVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DiabetesVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_var = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# GAN Models
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_vae(vae, train_loader, num_epochs=100):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = vae(batch)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(recon_batch, batch)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Total loss
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            total_loss += loss.item()
            
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def train_gan(generator, discriminator, train_loader, num_epochs=100, latent_dim=20):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            batch_size = batch.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1)
            label_fake = torch.zeros(batch_size, 1)
            
            output_real = discriminator(batch)
            d_loss_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, latent_dim)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_data)
            g_loss = criterion(output_fake, label_real)
            
            g_loss.backward()
            g_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

def generate_synthetic_data(original_data, model_type='vae', num_samples=1000):
    # Prepare data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data)
    
    # Convert to PyTorch dataset
    dataset = DiabetesDataset(scaled_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = scaled_data.shape[1]
    latent_dim = 20
    
    if model_type == 'vae':
        # Train VAE
        vae = DiabetesVAE(input_dim, latent_dim)
        train_vae(vae, train_loader)
        
        # Generate synthetic data using VAE
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim)
            synthetic_data = vae.decode(z)
            synthetic_data = synthetic_data.numpy()
    
    else:  # GAN
        # Train GAN
        generator = Generator(latent_dim, input_dim)
        discriminator = Discriminator(input_dim)
        train_gan(generator, discriminator, train_loader)
        
        # Generate synthetic data using GAN
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim)
            synthetic_data = generator(noise).numpy()
    
    # Inverse transform the scaled data
    synthetic_data = scaler.inverse_transform(synthetic_data)
    return synthetic_data

# Main execution
if __name__ == "__main__":
    # First generate base data using our existing function
    base_data = generate_diabetes_data(1000)
    
    # Select numerical columns for GAN/VAE training
    numerical_cols = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'HbA1c', 
                     'Fasting_Glucose', 'Random_Glucose', 'Total_Cholesterol',
                     'HDL', 'LDL', 'Triglycerides']
    
    # Generate synthetic data using both VAE and GAN
    vae_data = generate_synthetic_data(base_data[numerical_cols].values, model_type='vae')
    gan_data = generate_synthetic_data(base_data[numerical_cols].values, model_type='gan')
    
    # Create DataFrames with the synthetic data
    vae_df = pd.DataFrame(vae_data, columns=numerical_cols)
    gan_df = pd.DataFrame(gan_data, columns=numerical_cols)
    
    # Save the synthetic datasets
    vae_df.to_csv('synthetic_diabetes_vae.csv', index=False)
    gan_df.to_csv('synthetic_diabetes_gan.csv', index=False)
    
    # Visualize distributions
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(base_data[col], alpha=0.5, label='Original', bins=30)
        axes[idx].hist(vae_df[col], alpha=0.5, label='VAE', bins=30)
        axes[idx].hist(gan_df[col], alpha=0.5, label='GAN', bins=30)
        axes[idx].set_title(col)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.close() 