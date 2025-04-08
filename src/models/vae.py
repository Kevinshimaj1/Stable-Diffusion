import torch
import torch.nn as nn # fornisce i convolution layers e i pooling layers
import torch.nn.functional as F # importo le funzioni di nn come conv2d ecc.

class VAE(nn.Module): # Module è la classe base per tutte le reti neurali, ogni modello deve ereditare da questa classe (metodo forward)
    # definisco il costruttore
    def __init__(self, in_channels = 3, latent_channels = 4, image_size = 512):
        """
        Variational Autoencoder (VAE) per Stable Diffusion.

        Args:
            in_channels (int): numero di canali in input (ad es. 3 per RGB).
            latent_channels (int): numero di canali nello spazio latente.
            image_size (int): dimensione dell'immagine in input (es. 512 per 512x512).
            Per questa image_size occorrerà effettuare un preprocessing dei dati del dataset.
        """
        super(VAE, self).__init__ # invoco il costruttore __init__() della classe nn.Module

        # Calcolo la dimensione dello spazio latente
        self.latent_size = image_size // 8 # Fattore di downsampling = 8 (512 -> 64)
        self.latent_channels = latent_channels

        # === Definizione dell'Encoder ===
        # L'encoder mappa l'immagine in una distribuzione latente q(z | x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 4, stride = 2, padding = 1), # -> (64,256,256)
            nn.ReLU(), # se input >  0, restituisce l'input, altrimenti restituisce 0
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 64, 64)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # -> (512, 64, 64)
            nn.ReLU(),
        )

        # Mappo l'output dell'encoder in mu e logvar per la distribuzione latente
        self.fc_mu = nn.Conv2d(512, latent_channels, kernel_size=1)  # -> (latent_channels, 64, 64)
        self.fc_logvar = nn.Conv2d(512, latent_channels, kernel_size=1)  # -> (latent_channels, 64, 64)