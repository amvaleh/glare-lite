import os  
import argparse  
from typing import List, Tuple  

from PIL import Image  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms  
from torchvision.utils import save_image  


def pick_device() -> torch.device:  
    if torch.backends.mps.is_available():  
        return torch.device("mps")  
    return torch.device("cpu")  


class PairedFolderDataset(Dataset):  
    def __init__(self, low_dir: str, high_dir: str, size: int = 256):  
        self.low_dir = low_dir  
        self.high_dir = high_dir  
        self.size = size  
        self.files = self._collect_files()  
        self.tf = transforms.Compose([  
            transforms.Resize((size, size)),  
            transforms.ToTensor(),  
        ])  

    def _collect_files(self) -> List[str]:  
        low_files = set(os.listdir(self.low_dir))  
        high_files = set(os.listdir(self.high_dir))  
        common = sorted(list(low_files.intersection(high_files)))  
        common = [f for f in common if f.lower().endswith((".png", ".jpg", ".jpeg"))]  
        if len(common) == 0:  
            raise RuntimeError("No paired images found. Ensure filenames match in low/ and high/.")  
        return common  

    def __len__(self) -> int:  
        return len(self.files)  

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:  
        name = self.files[idx]  
        low_path = os.path.join(self.low_dir, name)  
        high_path = os.path.join(self.high_dir, name)  
        low_img = Image.open(low_path).convert("RGB")  
        high_img = Image.open(high_path).convert("RGB")  
        low = self.tf(low_img)  
        high = self.tf(high_img)  
        return low, high, name  


class SEBlock(nn.Module):  
    def __init__(self, ch: int, reduction: int = 8):  
        super().__init__()  
        hidden = max(4, ch // reduction)  
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)  
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        s = F.adaptive_avg_pool2d(x, 1)  
        s = F.relu(self.fc1(s), inplace=True)  
        s = torch.sigmoid(self.fc2(s))  
        return x * s  


class ConvBlock(nn.Module):  
    def __init__(self, in_ch: int, out_ch: int):  
        super().__init__()  
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)  
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  
        self.se = SEBlock(out_ch)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = F.relu(self.c1(x), inplace=True)  
        x = F.relu(self.c2(x), inplace=True)  
        x = self.se(x)  
        return x  


class GLARELiteUNet(nn.Module):  
    def __init__(self, base: int = 32):  
        super().__init__()  
        self.e1 = ConvBlock(3, base)  
        self.e2 = ConvBlock(base, base * 2)  
        self.e3 = ConvBlock(base * 2, base * 4)  
        self.pool = nn.MaxPool2d(2)  
        self.bottleneck = ConvBlock(base * 4, base * 4)  
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  
        self.d2 = ConvBlock(base * 4, base * 2)  
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)  
        self.d1 = ConvBlock(base * 2, base)  
        self.outc = nn.Conv2d(base, 3, 1)  
        self.tanh = nn.Tanh()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        e1 = self.e1(x)  
        p1 = self.pool(e1)  
        e2 = self.e2(p1)  
        p2 = self.pool(e2)  
        e3 = self.e3(p2)  
        b = self.bottleneck(e3)  
        u2 = self.up2(b)  
        c2 = torch.cat([u2, e2], dim=1)  
        d2 = self.d2(c2)  
        u1 = self.up1(d2)  
        c1 = torch.cat([u1, e1], dim=1)  
        d1 = self.d1(c1)  
        r = self.tanh(self.outc(d1))  
        y = torch.clamp(x + 0.5 * r, 0.0, 1.0)  
        return y  

def rgb_to_yuv(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r - 0.28886*g + 0.436*b
    v = 0.615*r - 0.51499*g - 0.10001*b
    return y, u, v

def color_loss(pred, target):
    _, up, vp = rgb_to_yuv(pred)
    _, ut, vt = rgb_to_yuv(target)
    return F.l1_loss(up, ut) + F.l1_loss(vp, vt)

def exposure_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  
    pred_gray = pred.mean(dim=1, keepdim=True)  
    tgt_gray = target.mean(dim=1, keepdim=True)  
    pred_mean = pred_gray.mean(dim=[2, 3])  
    tgt_mean = tgt_gray.mean(dim=[2, 3])  
    return F.l1_loss(pred_mean, tgt_mean)  


def saturation(x, eps=1e-6):
    mx, _ = x.max(dim=1, keepdim=True)
    mn, _ = x.min(dim=1, keepdim=True)
    return (mx - mn) / (mx + eps)

def sat_loss(pred, target):
    return F.l1_loss(saturation(pred), saturation(target))

def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:  
    model.train()  
    total = 0.0  
    n = 0  
    for low, high, _ in loader:  
        low = low.to(device)  
        high = high.to(device)  
        pred = model(low)  
        l1 = F.l1_loss(pred, high)  # cmnt: pixel fidelity loss
        lexp = exposure_loss(pred, high)  # cmnt: global exposure alignment loss

        lch = color_loss(pred, high) # cmnt: test 4 improvment
        lsat = sat_loss(pred, high) # cmnt: test 5 improvment

        # loss = l1 + 0.1 * lexp  # total loss (small weight on exposure)
        # loss = l1 + 0.1*lexp + 0.05*lch # test 4 improvment: adding a weight for colorloss
        loss = l1 + 0.05*lexp + 0.30*lch + 0.10*lsat # test 5 improvment: adding a weight for saturation loss

        opt.zero_grad(set_to_none=True)  
        loss.backward()  
        opt.step()  
        total += float(loss.detach().cpu().item())  
        n += 1  
    return total / max(1, n)  


@torch.no_grad()  
def validate_and_save_examples(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str, limit: int = 4) -> None:  
    model.eval()  
    os.makedirs(out_dir, exist_ok=True)  
    saved = 0  
    for low, high, names in loader:  
        low = low.to(device)  
        pred = model(low)  
        for i in range(pred.shape[0]):  
            if saved >= limit:  
                return  
            name = names[i]  
            grid = torch.cat([low[i].cpu(), pred[i].cpu(), high[i].cpu()], dim=2)  
            save_image(grid, os.path.join(out_dir, f"cmp_{saved:02d}_{name}"))  
            saved += 1  


@torch.no_grad()  
def enhance_single_image(model: nn.Module, device: torch.device, in_path: str, out_path: str, size: int = 512) -> None:  
    model.eval()  
    img = Image.open(in_path).convert("RGB")  
    tf = transforms.Compose([  
        transforms.Resize((size, size)),  
        transforms.ToTensor(),  
    ])  
    x = tf(img).unsqueeze(0).to(device)  
    y = model(x).squeeze(0).cpu()  
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  
    save_image(y, out_path)  


def main() -> None:  
    p = argparse.ArgumentParser()  
    p.add_argument("--train_dir", type=str, default="data/train", help="train folder with low/ and high/")  
    p.add_argument("--val_dir", type=str, default="data/val", help="val folder with low/ and high/")  
    p.add_argument("--epochs", type=int, default=5, help="number of epochs (start small)")  
    p.add_argument("--batch", type=int, default=4, help="batch size (reduce if memory issues)")  
    p.add_argument("--size", type=int, default=256, help="train resize (256 is fast)")  
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")  
    p.add_argument("--base", type=int, default=32, help="model width; 16 is faster, 32 is better")  
    p.add_argument("--out", type=str, default="runs/glare_lite", help="output folder for checkpoints/examples")  
    p.add_argument("--infer_in", type=str, default="", help="optional: single image path to enhance")  
    p.add_argument("--infer_out", type=str, default="runs/glare_lite/enhanced.png", help="optional: inference output path")  
    args = p.parse_args()  

    device = pick_device()  
    print("Device:", device)  

    model = GLARELiteUNet(base=args.base).to(device)  
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)  

    train_ds = PairedFolderDataset(os.path.join(args.train_dir, "low"), os.path.join(args.train_dir, "high"), size=args.size)  
    val_ds = PairedFolderDataset(os.path.join(args.val_dir, "low"), os.path.join(args.val_dir, "high"), size=args.size)  
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)  
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)  

    os.makedirs(args.out, exist_ok=True)  

    for epoch in range(1, args.epochs + 1):  
        loss = train_one_epoch(model, train_loader, opt, device)  
        print(f"Epoch {epoch}/{args.epochs} - loss: {loss:.5f}")  
        ckpt_path = os.path.join(args.out, f"ckpt_epoch_{epoch}.pt")  
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)  
        validate_and_save_examples(model, val_loader, device, os.path.join(args.out, "examples"), limit=4)  

    if args.infer_in:  
        enhance_single_image(model, device, args.infer_in, args.infer_out, size=max(args.size, 512))  

    print("Done. Check:", args.out)  


if __name__ == "__main__":  
    main()  
