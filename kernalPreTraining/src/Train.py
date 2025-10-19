import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import json
import os

# preload the download path
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")


# ======================================================
# 🧠 Model: Simple 5x5 CA kernel with 3 input/output channels
# ======================================================
class CAKernel(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5, delta=0.1):
        super().__init__()
        self.channels = in_channels
        self.delta = delta
        self.bestEval = None
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, padding_mode='circular', bias=False),
            nn.ReLU()
        )

    def forward(self, x, steps=10):
        for _ in range(steps):
            x = self.activate(x)
        return x

    def activate(self, x):
        x_updated = x + self.delta * self.model(x)
        return x_updated.clamp(0.0, 1.0)

    def exportToPlaygroundFormat(self, filepath="TrainedWeights.json"):
        weights = self.model[0].weight.detach().cpu().tolist()

        # Simplified activation: linear delta
        activation_code = f"""fn activation(convX: f32) -> f32 {{
    var lastX: f32 = activationContext.cellState[activationContext.channel];
    var lastXRelu: f32 = max(0.0, lastX);
    var xUpdated: f32 = lastXRelu + ({self.delta} * convX);
    return clamp(xUpdated, 0.0, 1.0);
}}"""

        export_dict = {
            "weights": weights,
            "activationCode": activation_code,
            "normalize": False
        }

        downloads_save__path = os.path.join(downloads_path, filepath)
        with open(downloads_save__path, "w") as f:
            json.dump(export_dict, f, indent=2)

        print(f"CA kernel exported to {downloads_save__path}")

    def saveBest(self, score):
        if self.bestEval is None or score < self.bestEval:
            self.bestEval = score
            self.best_weights = {k: v.clone() for k, v in self.state_dict().items()}
            print(f"New best model saved with score: {self.bestEval:.6f}")

    def loadBest(self):
        if self.bestEval is not None:
            self.load_state_dict(self.best_weights)
            print(f"Best model with score {self.bestEval:.6f} loaded.")
        else:
            print("No best model to load.")

# ======================================================
# ⚖️ Loss Function: Enhanced NCA Loss with Laplacian
# ======================================================
def ca_loss(output, target_diff,
            mse_weight=1.0):
    """
    Multi-term NCA loss:
    - mse: pixel-wise difference
    - var_reg: match variance to prevent collapse
    - total variation: smooth patterns
    - fft loss: encourage structural similarity in frequency domain
    - laplacian loss: match edges and structural features
    """
    # --- Pixel-wise loss ---
    mse = nn.MSELoss()(output, target_diff)

    # --- Combine all terms ---
    total = (mse_weight * mse)

    return total, mse


# ======================================================
# 🚀 Trainer Class
# ======================================================
class CAPretrainer:
    def __init__(self, img_path, img_size=128, device=None, lr=1e-3):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CAKernel().to(self.device)
        self.target = self._load_image(img_path, img_size, out_channels=self.model.channels).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.99))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200, eta_min=1e-4)

    def _load_image(self, path, size, out_channels=3):
        img = Image.open(path).convert("RGB")
        transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]

        c_in = img_tensor.shape[1]
        if out_channels > c_in:
            extra = torch.zeros((1, out_channels - c_in, size, size), device=img_tensor.device)
            img_tensor = torch.cat([img_tensor, extra], dim=1)

        return img_tensor
    
    def project_weights_zero_sum(self):
        """Hard-project all conv weights so each out-channel's total sum == 0."""
        with torch.no_grad():
            for layer in self.model.model:
                if isinstance(layer, nn.Conv2d):
                    w = layer.weight
                    # sum over in_channels, kernel_height, kernel_width for each out-channel
                    sums = w.sum(dim=(1, 2, 3), keepdim=True)
                    n = float(w.shape[1] * w.shape[2] * w.shape[3])
                    w.sub_(sums / n)

    def train(self,
            epochs=1000,
            max_steps=10,
            project_weights=True,
            print_every=5,
            lr=None,
            noise_scale=0.1):
        """
        Train the CA by computing loss at each automata step, adding noise to first 3 channels.
        Logs the best (lowest) MSE from all iterations per epoch.
        """
        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        target = self.target
        try: 
            for step in range(epochs):
                self.optimizer.zero_grad()

                # Random noise as initial state
                x = torch.rand_like(target)

                total_loss = 0.0
                total_mse = 0.0
                best_mse = float('inf')

                # Run automata for multiple steps, with noise injection
                for i in range(random.randint(1, max_steps)):
                    # Inject Gaussian noise only into the first 3 channels (e.g., RGB)
                    noise = torch.randn_like(x[:, :3]) * noise_scale
                    x = x.clone()
                    x[:, :3] = (x[:, :3] + noise).clamp(0.0, 1.0)

                    # One CA update
                    x = self.model.activate(x)

                    # Compute loss at this step
                    step_loss, mse = ca_loss(x, target)
                    total_loss = total_loss + step_loss
                    total_mse = total_mse + mse

                    # Track best (lowest) MSE
                    if mse.item() < best_mse:
                        best_mse = mse.item()

                # Average losses across steps
                total_loss /= max_steps
                total_mse /= max_steps

                self.model.saveBest(total_loss.item())

                # Backpropagation
                total_loss.backward()

                # Gradient clipping & optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # Projection to zero-sum weights
                if project_weights:
                    self.project_weights_zero_sum()

                # Logging
                if step % print_every == 0:
                    print(f"[{step}] Loss: {total_loss.item():.6f} | Avg MSE: {total_mse.item():.6f} | Best MSE: {best_mse:.6f}")
        except KeyboardInterrupt:
            print("Training interrupted. Skipping to evaluation.")

    def animate(self, steps=20, interval=10):
        """Animate CA evolution from random noise."""
        with torch.no_grad():
            x = torch.rand_like(self.target)
            frames = [x[:, :3].clone()]
            for _ in range(steps):
                x = self.model.activate(x)
                frames.append(x[:, :3].clone())

        fig, ax = plt.subplots()
        im = ax.imshow(frames[0][0].permute(1, 2, 0).cpu())
        ax.axis("off")

        def update(frame_idx):
            im.set_data(frames[frame_idx][0].permute(1, 2, 0).cpu())
            return [im]

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
        plt.show()

# ======================================================
# Usage
# ======================================================
if __name__ == "__main__":
    trainer = CAPretrainer("kernalPreTraining/trainingImages/Emoji.png", img_size=32)

    trainer.train(epochs=1000, max_steps=40)
    trainer.model.loadBest()

    save = input("Save pattern? ")
    if save.strip().lower() == 'y':
        trainer.model.exportToPlaygroundFormat()

    trainer.animate(steps=500)