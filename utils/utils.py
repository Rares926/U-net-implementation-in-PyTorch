
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import os
# Internal framework imports

# Typing imports imports

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=====> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=====> Loading checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])


def save_predictions(loader, model, folder="saved_images/", device="cuda",batch_size=1):
    model.eval() 
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds < 0.5).float() 
        torchvision.utils.save_image(preds, f"{folder}/preds/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/real/{idx}.png")

    if batch_size==1:
        show_visual_result_with_original_image("Z:\Xperi\semantic_segmentation\data/val_images","Z:\Xperi\semantic_segmentation\saved predictions/real","Z:\Xperi\semantic_segmentation\saved predictions\preds")
    else:
         show_visual_result("Z:\Xperi\semantic_segmentation\saved predictions/real","Z:\Xperi\semantic_segmentation\saved predictions\preds")

    model.train()

def show_visual_result_with_original_image(img,mask,pred):

    img_list = os.listdir(img)
    mask_list=os.listdir(mask)
    pred_list=os.listdir(pred)

    K=[img,mask,pred]
    Y=[img_list,mask_list,pred_list]

    rows=len(img_list)
    adder=1

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Results overview', fontsize=30)
    for idx in range(1,rows+1):
        for jdx in range(1,4):
            path=os.path.join(K[jdx-1],Y[jdx-1][idx-1])
            image=cv2.imread(path)
            fig.add_subplot(rows,3,adder)
            adder+=1
            plt.imshow(image)

    plt.show()


def show_visual_result(mask,pred):

    mask_list=os.listdir(mask)
    pred_list=os.listdir(pred)

    K=[mask,pred]
    Y=[mask_list,pred_list]

    rows=len(mask_list)
    adder=1

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Results overview', fontsize=30)
    for idx in range(1,rows+1):
        for jdx in range(1,3):
            path=os.path.join(K[jdx-1],Y[jdx-1][idx-1])
            image=cv2.imread(path)
            fig.add_subplot(2,rows,adder)
            adder+=1
            plt.imshow(image)

    plt.show()


