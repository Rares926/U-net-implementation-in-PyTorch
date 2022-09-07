
import torch


def check_dice_score(loader, model, device="cuda"):

    predicted = 0
    pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            preds = (preds[:,0] < 0.5).float()

            predicted += (preds == y).sum()
            pixels += torch.numel(preds)
                                    
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {predicted}/{pixels} with acc {predicted/pixels*100:.2f}")

    print(f"Dice score: {dice_score/len(loader)}")

    model.train()

