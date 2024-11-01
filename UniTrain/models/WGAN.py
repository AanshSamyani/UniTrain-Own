from torchsummary import summary
import torch.nn as nn
    

class Discriminator(nn.Module):

        channels_img = 3
        features_d = 64 
        
        
        def _block(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2),
            )

        
        disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            _block(features_d, features_d * 2, 4, 2, 1),
            _block(features_d * 2, features_d * 4, 4, 2, 1),
            _block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )



class Generator(nn.Module):
    
    channels_img = 3
    features_g = 64
    channels_noise = 1024
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
  
    gen = nn.Sequential(
        _block(channels_noise, features_g * 16, 4, 1, 0),  
        _block(features_g * 16, features_g * 8, 4, 2, 1),  
        _block(features_g * 8, features_g * 4, 4, 2, 1),  
        _block(features_g * 4, features_g * 2, 4, 2, 1),  
        nn.ConvTranspose2d(
            features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
        ),
        nn.Tanh(),
        )
