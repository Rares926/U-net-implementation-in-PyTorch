from torch.utils.data import DataLoader

#internal framework
from core.dataset import CarvanaDataset


def get_loaders(train_dir,train_maskdir,val_dir,val_maskdir,batch_size,train_transform,val_transform,num_workers=4,pin_memory=True,):

    train_ds = CarvanaDataset(image_dir=train_dir,mask_dir=train_maskdir,transform=train_transform)
    val_ds   = CarvanaDataset(image_dir=val_dir,mask_dir=val_maskdir,transform=val_transform)

    train_loader  = DataLoader(train_ds,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    val_loader    = DataLoader(val_ds,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=False)

    return train_loader, val_loader

