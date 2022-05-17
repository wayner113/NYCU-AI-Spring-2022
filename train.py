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

INPUT_FOLDER = "./training_input/"
IMAGE_FOLDER = INPUT_FOLDER + "images/"
OUTPUT_FOLDER = ""

class Train_Dataset:
    def __init__(self, data, transform=None, path="train_images/"):
        self.data = data
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        image_path = self.path + record["image_id"]
        image = np.array(cv2.imread(image_path)).astype(np.uint8)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return {"image" : image, "target" : record['hotel_id_code']}

class Train_Model(nn.Module):
    def __init__(self, classes=100, backbone_name="efficientnet_b0"):
        super(Train_Model, self).__init__()
        self.backbone = timm.create_model(backbone_name, num_classes=classes, pretrained=True)
    def forward(self, x):
        return self.backbone(x)
#---------------------------------------transform--------------------------------------------
base_transform = A.Compose([A.ToFloat(), APT.transforms.ToTensorV2() ])

train_transform = A.Compose([A.HorizontalFlip(p=0.75), A.VerticalFlip(p=0.25),
    A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT), A.OpticalDistortion(p=0.25), A.Perspective(p=0.25), 
    A.CoarseDropout(p=0.5, min_holes=1, max_holes=6, min_height=16, max_height=64, min_width=16,  max_width=64), 
    A.CoarseDropout(p=0.75, max_holes=1, min_height=64, max_height=128, min_width=64,  max_width=128, fill_value=(255,0,0)),
    A.RandomBrightnessContrast(p=0.75), A.ToFloat(), APT.transforms.ToTensorV2() ])

test_transform = A.Compose([A.CoarseDropout(p=0.75, max_holes=1, min_height=64, 
            max_height=128, min_width=64, max_width=128, fill_value=(255,0,0)), 
    A.ToFloat(), APT.transforms.ToTensorV2() ])
#------------------------------------end of transform----------------------------------------
train_csv = pd.read_csv(INPUT_FOLDER + "train.csv")
train_csv["hotel_id_code"] = train_csv["hotel_id"].astype('category').cat.codes.values.astype(np.int64)
id_code = train_csv.drop(columns=["image_id"]).drop_duplicates().reset_index(drop=True)
id_code.to_csv(OUTPUT_FOLDER + 'id_code mapping.csv', index=False)
train_dataset = Train_Dataset(train_csv, train_transform, path=IMAGE_FOLDER)

def save_checkpoint(model, scheduler, optimizer, epoch, loss=None, score=None):
    checkpoint = {"epoch": epoch, "model": model.state_dict(), "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(), "loss": loss, "score": score}
    torch.save(checkpoint, f"{OUTPUT_FOLDER}checkpoint.pt")

def train_epoch(model, loader, criterion, optimizer, scheduler, epoch):
    losses = []
    targets_all = []
    outputs_all = []
    model.train()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        optimizer.zero_grad()
        images = sample['image'].to('cpu')
        targets = sample['target'].to('cpu')
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        losses.append(loss.item())
        targets_all.extend(targets.cpu().numpy())
        outputs_all.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        score = np.mean(targets_all == np.argmax(outputs_all, axis=1))
        desc = f"Epoch {epoch}/5 - Train loss:{loss:0.4f}, Accuracy: {score:0.4f}"
        t.set_description(desc)
    return np.mean(losses), score

def test(loader, model):
    targets = []
    outputs = []
    model.eval()
    t = tqdm(loader)
    for i, sample in enumerate(t):
        images = sample['image'].to('cpu')
        target = sample['target'].to('cpu')
        output = model(images)
        targets.extend(target.cpu().numpy())
        outputs.extend(torch.sigmoid(output).detach().cpu().numpy())
    y = np.repeat([targets], repeats=5, axis=0).T
    preds = np.argsort(-np.array(outputs), axis=1)[:, :5]
    accuracy5 = (preds == y).any(axis=1).mean()
    accuracy1 = np.mean(targets == np.argmax(outputs, axis=1))
    print(f"Accuracy in one: {accuracy1:0.4f}, Accuracy in five: {accuracy5:0.4f}")

def train_test(dataframe):
    test_data = dataframe.groupby("hotel_id").sample(1, random_state=60)
    train_data = dataframe[~dataframe["image_id"].isin(test_data["image_id"])]
    train_dataset = Train_Dataset(train_data, train_transform, path=IMAGE_FOLDER)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=64, shuffle=True, drop_last=True)
    test_dataset = Train_Dataset(test_data, test_transform, path=IMAGE_FOLDER)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=64, shuffle=False)
    model = Train_Model(dataframe["hotel_id_code"].nunique(), "efficientnet_b0")
    model = model.to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=5,
                    steps_per_epoch=len(train_loader), div_factor=10, final_div_factor=1, pct_start=0.1, anneal_strategy="cos")
    for epoch in range(1, 6):
        train_loss, train_score = train_epoch(model, train_loader, criterion, optimizer, scheduler, epoch)
        save_checkpoint(model, scheduler, optimizer, epoch, train_loss, train_score)
        test(test_loader, model)

train_test(train_csv)