"""Grad-CAM explanations for the fine-tuned VGG16 (uses the `grad-cam` package)."""
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.image.data import MEAN, STD, build_transforms

# features[29] = ReLU after conv5_3 (last conv layer of VGG16), 14x14 feature map at 224 input.
TARGET_LAYER_INDEX = 29


class GradCAMExplainer:
    def __init__(self, model, class_names, device="cpu", img_size=224):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        # In-place ReLUs clash with the gradient hooks used by Grad-CAM.
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.class_names = list(class_names)
        self.tf = build_transforms(img_size, train=False)
        self.cam = GradCAM(model=self.model,
                           target_layers=[self.model.features[TARGET_LAYER_INDEX]])

    def _prepare(self, pil_img):
        x = self.tf(pil_img.convert("RGB")).unsqueeze(0)
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        rgb = (x[0] * std + mean).permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)
        return x.to(self.device), rgb  # rgb = exactly the crop the model sees

    @torch.no_grad()
    def predict(self, pil_img, topk=5):
        x, _ = self._prepare(pil_img)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        conf, idx = probs.topk(min(topk, len(self.class_names)))
        return [(self.class_names[i], float(c)) for c, i in zip(conf, idx)]

    def explain(self, pil_img, target_class=None):
        """Return prediction, confidence and a Grad-CAM overlay.

        target_class: class index to explain (default: the predicted class).
        """
        x, rgb = self._prepare(pil_img)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        pred = int(probs.argmax())
        cls = pred if target_class is None else int(target_class)
        heat = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)
        return {
            "pred": self.class_names[pred], "pred_idx": pred, "confidence": float(probs[pred]),
            "explained_class": self.class_names[cls], "explained_conf": float(probs[cls]),
            "image": (rgb * 255).astype(np.uint8), "heatmap": heat, "overlay": overlay,
        }
