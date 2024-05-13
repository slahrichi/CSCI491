import matplotlib.pyplot as plt
import os 
from PIL import Image
import pandas as pd
from torchvision import transforms, models
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import decoders
from base import resnet 
import metrics
import argparse

# 0. Parse args
parser = argparse.ArgumentParser(
    description="""This script loads a pretrained encoder, and trains a decoder on the
    SustainBench task""")

parser.add_argument(
    "--data", type=str, default=None,
    choices=['field_delineation'],
    required=False,
    help="""The training task to attempt. 
    Valid task include ["field delineation"]""",
)

parser.add_argument(
    "--data_folder", type=str, default=None,
    required=True,
    help="""path to benchmark data main folder.""",
)

parser.add_argument(
    "--encoder",
    type=str,
    default="resnet50",
    choices=["resnet50"],
    help="""The encoder to use."""
)

parser.add_argument(
    "--fine_tune_encoder",
    default=False,
    help="""A flag Whether to fine tune the encoder during
    supervision. If False, then gradients will not be calculated on the encoder. If True, then
    the gradients will be calculated.""",
)

parser.add_argument(
    "--decoder",
    type=str,
    default="unet",
    choices=["unet"],
    help="""The decoder to use. Valid inputs include ["unet"]. By default
    the decoder is 'unet' for segmentation. """,
)

parser.add_argument(
    "--lr", type=float, default=1e-3, help="The learning rate. Default 1e-3."
)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="""Whether to use the GPU. Default 'auto' which uses
    a GPU if they are available. It is recommended that you explicitly set the GPU using a value of
    'cuda:0' or 'cuda:1' so that you can more easily track the model.""",
)

parser.add_argument(
    "--criterion",
    type=str,
    default="softiouloss",
    choices=["softiouloss", "BCEWithLogitsLoss","crossentropy"],
    help="""Select the criterion to use. By default, the
    criterion is 'softiou' (stylized: SoftIoU), and this should be the default value for semantic
    segmentation tasks. Other options include crossentropy and BCEWithLogitsLossBinary Cross Entropy Loss.""",
)

parser.add_argument(
    "--accuracy",
    type=str,
    default="iou",
    choices=["iou", "f1", "top1precision"],
    help="""Select the accuracy metric to use. Valid inputs include ["iou", "f1", "top1precision"].
    By default, the criterion is 'iou' , and this should be the default value for semantic
    segmentation tasks. """,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="""Batch size for the data. Default is 16, as this
    was found to work optimally on the GPUs we experiment with.""",
)

parser.add_argument(
    "--epochs", 
    type=int,
    default=100, help="Number of epochs. Default 100."
)

parser.add_argument(
    "--pretrained", 
    default=False, help="Whether to load pretrained models"
)


class CropDelineationDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, split, transform=None, img_transform=None, mask_transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame['split'] == split]
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data_frame.iloc[idx, 1]) + '.jpeg') 
        mask_name = os.path.join(self.mask_dir, str(self.data_frame.iloc[idx, 1]) + '.png')  

        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')  
        
        if self.img_transform:
            image = self.img_transform(image)
            mask = self.mask_transform(mask)

        return image, mask.type(torch.LongTensor)


def main():
    args = parser.parse_args()

    # 1. Create custom DataLoader class

    if args.data == "field_delineation":
        mean = [0.2384, 0.2967, 0.3172]
        std = [0.1873, 0.1226, 0.1138]

    img_transform = transforms.Compose([
        # Add more here if needed, this dataset is already cropped and sized for us
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])

    mask_transform = transforms.Compose([
        # Add more here if needed, this dataset is already cropped and sized for us
        transforms.ToTensor()
    ])

    # 2. Instantiate dataloaders for train and test splits

    train_dataset = CropDelineationDataset(csv_file=os.path.join(args.data_folder, "clean_data.csv"),
                                img_dir=os.path.join(args.data_folder, "imgs"),
                                mask_dir=os.path.join(args.data_folder, "masks_filled"),
                                split='train',
                                transform=True,
                                img_transform=img_transform,
                                mask_transform=mask_transform)

    val_dataset = CropDelineationDataset(csv_file=os.path.join(args.data_folder, "clean_data.csv"),
                                img_dir=os.path.join(args.data_folder, "imgs"),
                                mask_dir=os.path.join(args.data_folder, "masks_filled"),
                                split='val',
                                transform=True,
                                img_transform=img_transform,
                                mask_transform=mask_transform)

    test_dataset = CropDelineationDataset(csv_file=os.path.join(args.data_folder, "clean_data.csv"),
                                img_dir=os.path.join(args.data_folder, "imgs"),
                                mask_dir=os.path.join(args.data_folder, "masks_filled"),
                                split='test',
                                transform=True,
                                img_transform=img_transform,
                                mask_transform=mask_transform)


    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size//2, shuffle=False, drop_last=True)


    # 3. Load pretrained resnet50 and unet 

    if args.encoder == "resnet50":
        pretrained_r50 = models.resnet50(pretrained=True)
        state_dict = pretrained_r50.state_dict()
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        encoder = resnet.resnet50(inter_features=True)
        encoder.load_state_dict(state_dict, strict=True)

    if args.decoder == "unet":
        decoder = decoders.load(decoder_name="unet", encoder=encoder) 

    if args.fine_tune_encoder:
        # Chain the iterators to combine them.
        params = list(encoder.parameters())+list(decoder.parameters())
    else:
        params = decoder.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = metrics.load(metric_name = args.criterion, device=args.device)
    acc_metric = metrics.load(metric_name = args.accuracy, device=args.device)

    encoder.to(args.device)
    decoder.to(args.device)
    criterion.to(args.device)
    
    if not args.pretrained:

        if args.fine_tune_encoder:
            encoder.train()
        else:
            encoder.eval()
            print("Training with frozen encoder")
        decoder.train()

        # Training loop!
        print("Begin training...")
        for epoch in tqdm(range(args.epochs)):
            running_loss = 0.0
            running_acc = 0.0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(args.device)
                masks = masks.to(args.device)
                
                #Forward pass
                if args.fine_tune_encoder:
                    outputs = encoder(images)
                else:
                    with torch.no_grad():
                        outputs = encoder(images)
                outputs = decoder(outputs)
                loss = criterion(outputs, masks)
                
                pred_logits = torch.sigmoid(outputs)  
                pred_masks = (pred_logits > 0.5).long() # Convert to binary mask

                acc = acc_metric(pred_masks, masks)

                if batch_idx % 10 == 0:
                    print(f"\t Train Loss: {loss.item():.4f}. Train Accuracy {acc.item():.4f}")
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                running_loss += loss.item()
                running_acc += acc.item()

                optimizer.step()

            # when done with all batches, compute epoch stats
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = running_acc / len(train_loader)

            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}; Acc: {epoch_acc:.4f}")
        
        torch.save(encoder.state_dict(), "encoder.pt")
        torch.save(decoder.state_dict(), "decoder.pt")

    # Testing
    if args.pretrained:
        print("Loading saved models")
        encoder.load_state_dict(torch.load("encoder.pt"))
        decoder.load_state_dict(torch.load("decoder.pt"))

    encoder.eval()
    decoder.eval()
    print("Begin testing...")
    for batch_idx, (images, masks) in enumerate(test_loader):
        running_loss, running_acc = 0, 0
        images = images.to(args.device)
        masks = masks.to(args.device)

        outputs = encoder(images)
        outputs = decoder(outputs)
        loss = criterion(outputs, masks)

        pred_logits = torch.sigmoid(outputs) 
        pred_masks = (pred_logits > 0.5).long() # Convert to binary mask

        acc = acc_metric(pred_masks,masks)

        running_loss += loss.item() / images.size(0)
        running_acc += acc.item() / images.size(0)

        if batch_idx % 10 == 0:
            print(f"\t Test Loss: {loss.item():.4f}, Test Accuracy: {acc.item():.4f}")
        

if __name__ == "__main__":
    main()