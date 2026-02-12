
import os                              
import pandas as pd                    
from PIL import Image                    
import torch                           
from torch.utils.data import Dataset, DataLoader  



class CYPImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        csv_file: path to CSV with columns [Drug_ID, Drug, Y]
        image_dir: base folder containing PNGs named {Drug_ID}_{Y}.png
        transform: torchvision transforms to apply
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # 1) Read row
        row = self.df.iloc[idx]
        # Cast Drug_ID and Y to integer to match your filename pattern:
        drug_id = int(row['Drug_ID'])  
        label   = int(row['Y'])         
        

        # 2) Build image path
        img_fp  = os.path.join(
            self.image_dir,
            f"{drug_id}_{label}.png"    
        )
        # 3) Load image
        image   = Image.open(img_fp).convert('RGB')
        # 4) Apply transforms (resize, to-tensor, normalize)
        if self.transform:
            image = self.transform(image)
        # 5) Return (image_tensor, label_tensor)
        return image, torch.tensor(float(label), dtype=torch.float32)
        
        



