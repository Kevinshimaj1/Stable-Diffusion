import torch
import torch.nn as nn # fornisce i convolution layers e i pooling layers
import torch.nn.functional as F # importo le funzioni di nn come conv2d ecc.

class VAE(nn.Module): # Module è la classe base per tutte le reti neurali, ogni modello deve ereditare da questa classe (metodo forward)
    # definisco il costruttore
    def __init__(self, in_channels=3, latent_channels=4, image_size=512):
        """
        Variational Autoencoder (VAE) per Stable Diffusion.

        Args:
            in_channels (int): numero di canali in input (ad es. 3 per RGB).
            latent_channels (int): numero di canali nello spazio latente.
            image_size (int): dimensione dell'immagine in input (es. 512 per 512x512).
            Per questa image_size occorrerà effettuare un preprocessing dei dati del dataset.
        """
        super(VAE, self).__init__() # invoco il costruttore __init__() della classe nn.Module

        # Controlla che image_size sia un multiplo di 8
        if image_size % 8 != 0:
            raise ValueError("image_size deve essere un multiplo di 8 per il fattore di downsampling")

        # Calcolo la dimensione dello spazio latente
        self.latent_size = image_size // 8 # Fattore di downsampling = 8 (512 -> 64)
        self.latent_channels = latent_channels

        # === Definizione dell'Encoder ===
        # L'encoder mappa l'immagine in una distribuzione latente q(z | x)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # -> (64,256,256)
            nn.ReLU(), # se input >  0, restituisce l'input, altrimenti restituisce 0
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 128, 128)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 64, 64)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # -> (512, 64, 64)
            nn.ReLU(),
        )

        # == Definizione del vettore latente z == 
        # Mappo l'output dell'encoder in mu e logvar per la distribuzione latente
        self.fc_mu = nn.Conv2d(512, latent_channels, kernel_size=1)  # -> (latent_channels, 64, 64), dove latent_channels = 512
        self.fc_logvar = nn.Conv2d(512, latent_channels, kernel_size=1)  # -> (latent_channels, 64, 64), ""

        # == Definizione del Decoder ==
        # Il decoder costruisce l'immagine a partire da z
        self.decoder = nn.Sequential(
            # il primo convtranspose2d serve solo per diminuire i latent_channels a 3 (prima erano a 4)
            nn.ConvTranspose2d(latent_channels, 512, kernel_size=3, stride=1, padding=1), # serve a iniziare il processo di decodifica espandendo il numero di feature (canali) della rappresentazione latente ( z ), preparando i successivi layer di upsampling
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 128, 128)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 256, 256)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64, 512, 512)
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1),  # -> (in_channels, 512, 512): immagine ricostruita
            nn.Tanh(),  # Output normalizzato in [-1, 1] tramite la tangente iperbolica, introduce non linearità
        )

    def encode(self, x):
        """
        Codifica l'immagine in una distribuzione latente q(z | x)

        Args:
            x (torch.Tensor): Immagine in input, shape (batch_size, in_channels, image_size, image_size)
        
        Returns:
            mu (torch.Tensor): Media della distribuzione latente.
            logvar (torch.Tensor): Logaritmo della varianza della distribuzione latente.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    # questo trick permette di effettuare il backpropagation nella rete e rende l'equazione differenziabile
    def reparameterization(self, mu, logvar):
        """
        Applica il reparameterization trick per campionare z in modo differenziabile.

        Args: 
            mu (torch.Tensor): Media della distribuzione latente.
            logvar (torch.Tensor): Logaritmo della varianza della distribuzione latente.
        
        Returns:
            z (torch.Tensor): Campione dell spazio latente.
        """
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std) # Campiono epsilon dalla distribuzione normale N(0,1)
        z = mu + eps * std # z = mu + sigma * epsilon
        return z
    
    def decode(self, z):
        """
        Ricostruisce l'immagine a partire da un campione z dello spazio latente.

        Args:
            z (torch.Tensor): Campione dell ospazio latente, shape (batch_size, latent_channels, latent_size, latent_size)
        
        Returns:
            x_recon (torch.Tensor): Immagine ricostruita.
        """
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x):
        """
        Passaggio completo attraverso il VAE: codifica, campionamento e decodifica.

        Args:
            x (torch.Tensor): immagine in input.
        
        Returns:
            x_recon (torch.Tensor): Immagine ricostruita.
            mu (torch.Tensor): Media della distribuzione latente.
            logvar (torch.Tensor): Logaritmo della varianza della distribuzione latente.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """
        Genera nuove immagini campionando dallo spazio latente.

        Args:
            num_sample (int): Numero di immagini da generare.
            device (str): Dispositivo su cui eseguire il campionamento.

        Returns:
            samples (torch.Tensor): Immagini generate.
        """
        z = torch.randn(num_samples, self.latent_channels, self.latent_size, self.latent_size).to(device)
        samples = self.decode(z)
        return samples

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """
    Calcola la loss del VAE: errore di ricostruzione + divergenza KL

    Args:
        x (torch.Tensor): Immagine originale.
        x_recon (torch.Tensor): Immagine ricostruita.
        mu (torch.Tensor): Media della distribuzione latente.
        logvar (torch.Tensor): Logaritmo della varianza della distribuzione latente.
        beta (float): Peso per la divergenza KL (beta-VAE).
    """
    # Errore di ricostruzione. Mean-Squared-Error
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")

    # Divergenza KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Loss totale
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss

# Test
if __name__ == "__main__":
    # Parametri
    in_channels = 3
    latent_channels = 4
    image_size = 512
    batch_size = 4
    
    # Creo un'istanza del VAE
    vae = VAE(in_channels, latent_channels, image_size)
    
    # Creo un input di esempio
    x = torch.randn(batch_size, in_channels, image_size, image_size)
    
    # Passaggio in avanti
    x_recon, mu, logvar = vae(x)
    
    # Calcolo la loss
    loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar)
    
    print(f"Loss: {loss.item()}")
    print(f"Reconstruction Loss: {recon_loss.item()}")
    print(f"KL Divergence: {kl_loss.item()}")