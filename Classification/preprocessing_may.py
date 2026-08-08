import os
from PIL import Image
from glob import glob
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from swin_custom import SwinUnet

# image_input_dir = "./archive/brisc2025/classification_task/train/glioma"

# image_output_dir = "./archive/brisc2025/classification_input/glioma" # folder to save resized images
 

# # Desired output size
# new_size = (448, 448)   # (width, height)

# # Create output directories if they don't exist
# os.makedirs(image_output_dir, exist_ok=True)


# def resize_image(input_path, output_path, size, is_mask=False):
#     img = Image.open(input_path)

#     # For masks → nearest neighbor (keeps integer class values)
#     if is_mask:
#         img = img.resize(size, Image.NEAREST)
#     else:
#         # For images → bilinear interpolation
#         img = img.resize(size, Image.BILINEAR)

#     img.save(output_path)

# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

 

# print("Done! GL Images resized and saved.")


# image_input_dir = "./archive/brisc2025/classification_task/train/pituitary"          # folder containing original .jpg images
 
# image_output_dir = "./archive/brisc2025/classification_input/pituitary" # folder to save resized images

# os.makedirs(image_output_dir, exist_ok=True)

# # Resize all JPG images
# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! PI Images and masks resized and saved.")

# image_input_dir = "./archive/brisc2025/classification_task/train/meningioma"          # folder containing original .jpg images
#            # folder containing original .png masks

# image_output_dir = "./archive/brisc2025/classification_input/meningioma" # folder to save resized images
 
# os.makedirs(image_output_dir, exist_ok=True)
 
# # Resize all JPG images
# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! ME Images and masks resized and saved.")

# image_input_dir = "./archive/brisc2025/classification_task/train/no_tumor"          # folder containing origi

# image_output_dir = "./archive/brisc2025/classification_input/no_tumor" # folder to save resized images
 
# os.makedirs(image_output_dir, exist_ok=True)

# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)
 

# print("Done! Images and masks resized and saved.")

# #-------------------------------------------------------------------Test dataset-----------------------------------------------------------------------------------

# image_input_dir = "./archive/brisc2025/classification_task/test/glioma"         
# image_output_dir = "./archive/brisc2025/classification_test/glioma/images" 

# os.makedirs(image_output_dir, exist_ok=True)

# # Resize all JPG images
# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! Images and masks resized and saved.")

# image_input_dir = "./archive/brisc2025/classification_task/test/meningioma"         

# image_output_dir = "./archive/brisc2025/classification_test/meningioma/images" # folder to save resized images

# os.makedirs(image_output_dir, exist_ok=True)

# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! Images and masks resized and saved.")


# image_input_dir = "./archive/brisc2025/classification_task/test/no_tumor"         
# image_output_dir = "./archive/brisc2025/classification_test/no_tumor/images" 

# os.makedirs(image_output_dir, exist_ok=True)
 
# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! Images and masks resized and saved.")

# image_input_dir = "./archive/brisc2025/classification_task/test/pituitary"          # folder containing original .jpg images
# image_output_dir = "./archive/brisc2025/classification_test/pituitary/images" # folder to save resized images

# os.makedirs(image_output_dir, exist_ok=True)
# for filename in os.listdir(image_input_dir):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(image_input_dir, filename)
#         output_path = os.path.join(image_output_dir, filename)
#         resize_image(input_path, output_path, new_size, is_mask=False)
#         print("Resized image:", filename)

# print("Done! Images and masks resized and saved.")

 


class PerImageZScore(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
    def apply(self, img, **params):
        img = img.astype(np.float32)
        m = img.mean()
        s = img.std()
        if s < 1e-6:
            s = 1e-6
        return (img - m) / s


#RESUNET MODEL CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

class GradCAM:
    """
    Fixes applied:
      1. Hooks registered once at construction, not per call.
      2. generate() runs its OWN forward pass WITHOUT no_grad so
         gradients actually flow.
      3. Model output is a tuple — we unpack and use only `final`.
      4. None-guard on gradients before use.
      5. autocast wraps only the forward inside generate(), not
         combined with no_grad.
    """
 
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
 
        # Register hooks once
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
 
    def _forward_hook(self, module, inp, out):
        # out is the activation tensor at this layer
        self.activations = out.detach()   # detach so we don't hold the graph
 
    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] is the gradient w.r.t. the layer's OUTPUT
        self.gradients = grad_out[0].detach()
 
    def generate(self, x):

        self.gradients = None
        self.activations = None

        self.model.zero_grad()

        device_type = "cuda" if x.is_cuda else "cpu"

        with torch.amp.autocast(device_type):

            final = self.model(x)

        prob = torch.sigmoid(final)

        score = (prob * (prob > 0.5)).sum()

        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "GradCAM hooks captured nothing."
            )

        grads = self.gradients
        acts = self.activations

        # =====================================================
        # CNN FEATURE MAP CASE
        # Shape: (B,C,H,W)
        # =====================================================
        if grads.dim() == 4:

            weights = grads.mean(dim=(2, 3), keepdim=True)

            cam = (weights * acts).sum(dim=1)

        # =====================================================
        # SWIN TRANSFORMER TOKEN CASE
        # Shape: (B,N,C)
        # =====================================================
        elif grads.dim() == 3:

            # Global average over tokens
            weights = grads.mean(dim=1, keepdim=True)

            # Weighted token activations
            cam = (weights * acts).sum(dim=2)

            # Reshape tokens -> spatial map
            B, N = cam.shape

            H = W = int(np.sqrt(N))

            cam = cam.reshape(B, H, W)

        else:
            raise RuntimeError(
                f"Unsupported gradient shape: {grads.shape}"
            )

        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
    # def generate(self, x):
    #     """
    #     x : preprocessed patch tensor, already on device.
    #     Returns a (H, W) numpy array in [0, 1].
    #     """
    #     self.gradients   = None
    #     self.activations = None
 
    #     self.model.zero_grad()
 
    #     # ── FIX 1: NO torch.no_grad() here — gradients must flow ──
    #     # ── FIX 2: autocast is fine around forward, but grad must be enabled ──
    #     device_type = "cuda" if x.is_cuda else "cpu"
    #     with torch.amp.autocast(device_type):
    #         final = self.model(x)
 
    #     # Sigmoid on the final logit map
    #     prob = torch.sigmoid(final)       # shape (1, 1, H, W)
 
    #     # Score = sum of confident foreground predictions
    #     # (equivalent to mean — just needs a scalar to differentiate)
    #     score = (prob * (prob > 0.5)).sum()
    #     score.backward()
 
    #     # ── FIX 4: guard against None gradients ──
    #     if self.gradients is None or self.activations is None:
    #         raise RuntimeError(
    #             "GradCAM hooks captured nothing. "
    #             "Check that the target layer is actually used in the forward pass."
    #         )
 
    #     grads = self.gradients   # (1, C, h, w)
    #     acts  = self.activations # (1, C, h, w)
 
    #     # Global-average-pool the gradients → channel weights
    #     weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
 
    #     # Weighted combination of activations
    #     cam = (weights * acts).sum(dim=1)               # (1, h, w)
    #     cam = F.relu(cam).squeeze().cpu().numpy()       # (h, w)
 
    #     # Normalize to [0, 1]
    #     cam_min, cam_max = cam.min(), cam.max()
    #     if cam_max - cam_min > 1e-8:
    #         cam = (cam - cam_min) / (cam_max - cam_min)
    #     else:
    #         cam = np.zeros_like(cam)
 
    #     return cam
# ------------------ Device ------------------
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

# ==============================
# LOAD MODEL (FIXED)
# ==============================
model = SwinUnet().to(device)

ckpt = torch.load(
    "SwinHafNet_model_8535.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

target_layer = model.swin_unet.layers_up[-1]
grad_cam = GradCAM(model, target_layer)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_mask_and_gradcam_from_dir(input_dir,
                                   mask_out_dir,
                                   grad_out_dir,
                                   model,
                                   gradcam,
                                   img_exts=('.jpg', '.jpeg', '.png'),
                                   threshold=0.5,
                                   resize_to=None):

    ensure_dir(mask_out_dir)
    ensure_dir(grad_out_dir)

    image_paths = []
    for ext in img_exts:
        image_paths += glob(os.path.join(input_dir, f'*{ext}'))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print('❌ No images found in', input_dir)
        return

    device = next(model.parameters()).device

    print(f"🚀 Processing {len(image_paths)} images...")

    for p in image_paths:
        name = Path(p).name
        name_stem = Path(p).stem

        # =========================
        # LOAD IMAGE
        # =========================
        pil = Image.open(p).convert('L')
        img_np = np.array(pil).astype(np.float32) / 255.0

        if resize_to is not None:
            img_disp = cv2.resize((img_np * 255).astype(np.uint8), resize_to)
            img_for_model = cv2.resize(img_np, resize_to)
        else:
            img_disp = (img_np * 255).astype(np.uint8)
            img_for_model = img_np

        # =========================
        # NORMALIZATION
        # =========================
        transformed = PerImageZScore(p=1.0).apply(img_for_model)

        x = torch.from_numpy(transformed).unsqueeze(0).unsqueeze(0).float().to(device)

        # =========================
        # 🔥 GRAD-CAM
        # =========================
        cam_map = gradcam.generate(x)

        # =========================
        # 🔥 PREDICTION
        # =========================
        with torch.no_grad():
            final = model(x)

            out_up = F.interpolate(
                final,
                size=img_for_model.shape,
                mode='bilinear',
                align_corners=False
            )

            prob_map = torch.sigmoid(out_up).squeeze().cpu().numpy()

        # =========================
        # NORMALIZE CAM
        # =========================
        cam_map = cam_map - cam_map.min()
        if cam_map.max() > 0:
            cam_map = cam_map / cam_map.max()

        cam_map = cv2.resize(
            cam_map,
            (prob_map.shape[1], prob_map.shape[0])
        )

        # =========================
        # MASK GENERATION
        # =========================
        bin_mask = (prob_map >= threshold).astype(np.float32)
        bin_mask = (bin_mask * 255).astype(np.uint8)

        # =========================
        # SAVE MASK (🔥 SAME NAME)
        # =========================
        mask_save_path = os.path.join(mask_out_dir, f'{name_stem}.png')
        cv2.imwrite(mask_save_path, bin_mask)

        # =========================
        # CREATE GRADCAM OVERLAY
        # =========================
        cam_uint8 = (cam_map * 255).astype(np.uint8)
        cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        overlay = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)
        overlay = cv2.resize(overlay, (cam_color.shape[1], cam_color.shape[0]))

        overlay_vis = cv2.addWeighted(overlay, 0.6, cam_color, 0.4, 0)

        # =========================
        # SAVE GRADCAM (🔥 SAME NAME)
        # =========================
        gradcam_save_path = os.path.join(grad_out_dir, f'{name_stem}.png')
        cv2.imwrite(gradcam_save_path, overlay_vis)

        print(f'✅ {name} -> mask + gradcam saved')

    print("🎯 DONE: All images processed successfully")


# input_dir = './archive/brisc2025/classification_input/glioma'
# mask_out_dir = './archive/brisc2025/classification_input/glioma_masks'
# grad_out_dir = './archive/brisc2025/classification_input/glioma_gradcam'

# save_mask_and_gradcam_from_dir(
#     input_dir,
#     mask_out_dir,
#     grad_out_dir,
#     model,
#     grad_cam,
#     resize_to=None,
#     threshold=0.3
# )

# input_dir = './archive/brisc2025/classification_input/meningioma'
# mask_out_dir = './archive/brisc2025/classification_input/meningioma_masks'
# grad_out_dir = './archive/brisc2025/classification_input/meningioma_gradcam'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.5)

# input_dir = './archive/brisc2025/classification_input/pituitary'
# mask_out_dir = './archive/brisc2025/classification_input/pituitary_masks'
# grad_out_dir = './archive/brisc2025/classification_input/pituitary_gradcam'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.5)

# input_dir = './archive/brisc2025/classification_input/no_tumor'
# mask_out_dir = './archive/brisc2025/classification_input/no_tumor_masks'
# grad_out_dir = './archive/brisc2025/classification_input/no_tumor_gradcam'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.5)

# # ----------------------------------------Test Dataset----------------------------------------------------

# input_dir = './archive/brisc2025/classification_test/glioma/images'
# mask_out_dir = './archive/brisc2025/classification_test/glioma/masks'
# grad_out_dir = './archive/brisc2025/classification_test/glioma/gradcams'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.3)

 

# input_dir = './archive/brisc2025/classification_test/meningioma/images'
# mask_out_dir = './archive/brisc2025/classification_test/meningioma/masks'
# grad_out_dir = './archive/brisc2025/classification_test/meningioma/gradcams'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.5)

# -----------------------
# Example usage (customize paths)
# -----------------------
# input_dir = './archive/brisc2025/classification_test/no_tumor/images'
# mask_out_dir = './archive/brisc2025/classification_test/no_tumor/masks'
# grad_out_dir = './archive/brisc2025/classification_test/no_tumor/gradcams'

# # If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
# save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.95)

input_dir = './archive/brisc2025/classification_test/pituitary/images'
mask_out_dir = './archive/brisc2025/classification_test/pituitary/masks'
grad_out_dir = './archive/brisc2025/classification_test/pituitary/gradcams'

# If your model expects fixed size (e.g., 224x224) set resize_to=(224,224), else None
save_mask_and_gradcam_from_dir(input_dir, mask_out_dir, grad_out_dir, model, grad_cam, resize_to=None, threshold=0.2)