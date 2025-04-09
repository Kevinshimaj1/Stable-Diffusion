import os
import torch
from torch.utils.data import Dataset
from PIL import Image # permette di aprire e manipolare file di immagini di diverso formato
import pandas as pd

class CelebADataset(Dataset):
    """
    Dataset personalizzato per caricare le immagini di CelebA.

    Args:
        root_dir (str): Directory radice del dataset CelebA (es. 'data/raw/celeba/').
        partition_file (str): Percorso al file di partizione (es. 'list_eval_partition.txt').
        partition (int): Partizione da caricare (0 = training, 1 = validation, 2 = test).
        transform (callable, optional): Trasformazioni da applicare alle immagini.
    """
    def __init__(self, root_dir, partition_file, partition=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "img_align_celeba")

        # Verifico che la directory delle immagini esista
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"La directory {self.image_dir} non esiste. Assicurati di aver scaricato il dataset CelebA.")

        # Carico il file di partizione
        self.partition_df = pd.read_csv(partition_file, delim_whitespace=True, header=None, skiprows=1, names=["image_id", "partition"])

        # Filtro le immagini in base alla partizione (0 = train, 1 = val, 2 = test)
        self.image_list = self.partition_df[self.partition_df["partition"] == partition]["image_id"].tolist()

        # Verifico che ci siano immagini nella partizione selezionata
        if len(self.image_list) == 0:
            raise ValueError(f"Nessuna immagine trovata per la partizione {partition}.")

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.

        Returns:
            int: Numero di immagini.
        """
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Carica e restituisce l'immagine all'indice specificato.

        Args:
            idx (int): Indice dell'immagine.

        Returns:
            torch.Tensor: Immagine come tensore.
        """
        # Costruisco il percorso dell'immagine
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Carico l'immagine con PIL
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Errore nel caricamento dell'immagine {img_path}: {str(e)}")

        # Applico le trasformazioni, se specificate
        if self.transform is not None:
            image = self.transform(image)

        return image

def get_dataloader(root_dir, partition_file, partition=0, transform=None, batch_size=16, shuffle=True, num_workers=4):
    """
    Crea un DataLoader per il dataset CelebA.

    Args:
        root_dir (str): Directory radice del dataset CelebA.
        partition_file (str): Percorso al file di partizione.
        partition (int): Partizione da caricare (0 = training, 1 = validation, 2 = test).
        transform (callable, optional): Trasformazioni da applicare alle immagini.
        batch_size (int): Dimensione del batch.
        shuffle (bool): Se True, mescola i dati.
        num_workers (int): Numero di worker per il caricamento dei dati.

    Returns:
        torch.utils.data.DataLoader: DataLoader per il dataset.
    """
    dataset = CelebADataset(root_dir, partition_file, partition, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Ottimizzo il trasferimento dei dati alla GPU
    )
    return dataloader

if __name__ == "__main__":
    # Test del dataset
    root_dir = "data/raw/celeba"
    partition_file = os.path.join(root_dir, "list_eval_partition.txt")

    # Creo il dataset (senza trasformazioni per ora)
    dataset = CelebADataset(root_dir, partition_file, partition=0, transform=None)

    # Stampo alcune informazioni
    print(f"Numero di immagini nel dataset (training): {len(dataset)}")

    # Carico la prima immagine
    image = dataset[0]
    print(f"Dimensione dell'immagine: {image.size}")