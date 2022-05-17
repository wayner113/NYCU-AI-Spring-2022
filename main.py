import os
import cv2 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
import albumentations as A
import albumentations.pytorch as APT
from tqdm import tqdm

IMG_SIZE = 256
INPUT_FOLDER = "./main_input/"
TEST_FOLDER = INPUT_FOLDER + "test_images/"

class Image_Dataset:
    def __init__(self, data, transform=None, folder="train_images/"):
        self.data = data
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.folder + record["image_id"]
        image = np.array(img_process(image_path)).astype(np.uint8)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return {  "image" : image  }
        
def img_process(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w, h, c = np.shape(img)
    if w > h:
        width = int((w - h) / 2)
        img = cv2.copyMakeBorder(img, width, width, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        width = int((h - w) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, width, width, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

class Train_Model(nn.Module):
    def __init__(self, n_classes=100, backbone_name="efficientnet_b0"):
        super(Train_Model, self).__init__()
        self.backbone = timm.create_model(backbone_name, num_classes=n_classes, pretrained=False)

    def forward(self, x):
        return self.backbone(x)

def predict(loader, model, num=5):
    preds = []
    with torch.no_grad():
        t = tqdm(loader)
        for i, sample in enumerate(t):
            input = sample['image'].to(parameter.device)
            outputs = model(input)
            outputs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds.extend(outputs)
    preds = np.argsort(-np.array(preds), axis=1)[:, :num]
    return preds

test_data = pd.DataFrame(data={"image_id": os.listdir(TEST_FOLDER), "hotel_id": ""}).sort_values(by="image_id")
hotel_id_code_df = pd.read_csv('./training/id_code mapping.csv')
id_code_map = hotel_id_code_df.set_index('hotel_id_code').to_dict()["hotel_id"]

class parameter:
    batch_size = 64
    workers = 0
    n_classes = hotel_id_code_df["hotel_id"].nunique()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(backbone_name, checkpoint_path, parameter):
    model = Train_Model(parameter.n_classes, backbone_name)
    checkpoint = torch.load(checkpoint_path, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    model = model.to(parameter.device)
    return model

base_transform = A.Compose([
    A.ToFloat(),
    APT.transforms.ToTensorV2(),])

test_dataset = Image_Dataset(test_data, base_transform, folder=TEST_FOLDER)
test_loader = DataLoader(test_dataset, num_workers=parameter.workers, batch_size=parameter.batch_size, shuffle=False)

model = get_model("efficientnet_b0", "./training/checkpoint.pt", parameter)
preds = predict(test_loader, model)
#transform the format of prediction into string 
preds = [[id_code_map[code] for code in classes] for classes in preds]
test_data["hotel_id"] = [str(list(l)).strip("[]").replace(",", "") for l in preds]

test_data.to_csv("submission.csv", index=False)
test_data.head()