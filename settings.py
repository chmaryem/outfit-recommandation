from pathlib import Path
import torch


BUNDLE_PATH = Path("outfit_reco.pth")
IMG_DIR = Path("C:/Users/Asus/Desktop/polyvore_outfits/images")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STATIC_ROUTE = "/images"          
STATIC_DIR = IMG_DIR             
TOPK_POOL = 30           
IMG_SIZE = 224
CSV_PATH = Path(r"C:/Users/Asus/Desktop/polyvore_outfits/augmented_items2.csv")  

