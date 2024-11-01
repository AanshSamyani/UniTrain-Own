from torchsummary import summary
import torch.nn as nn
    

class Discriminator(nn.Module):

        in_features = 512
        
        disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )



class Generator(nn.Module):
    
        latent_dim = 1
        img_dim = 512
        
        gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  
        )

