#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

from model.train import CLASSES

# Matching preprocess to train TRANSFORM_
PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Device to run model inference
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}\n')


def load_model():
    from model.train import MODEL_PT
    net = mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.last_channel, 3)

    state = torch.load(MODEL_PT, map_location=DEVICE)
    net.load_state_dict(state)
    net.to(DEVICE).eval()
    return net


def image_to_batch(image_path: str):
    img = Image.open(image_path).convert('RGB')
    x = _img_to_x(img)
    return x.to(DEVICE)


def frame_to_batch(frame_bgr: np.ndarray):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    x = _img_to_x(img)
    return x.to(DEVICE)


@torch.no_grad()
def infer(net, x):
    logits = net(x)  # (B, 3, H, W)
    probs = logits.softmax(dim=1)[0]
    pred_label = int(probs.argmax().item())
    pred_prob = probs[pred_label].item()
    return CLASSES[pred_label], pred_prob


def _img_to_x(img: Image):
    x = PREPROCESS(img)  # (3, H, W)
    x = x.unsqueeze(0)  # (1, 3, H, W)
    return x


if __name__ == '__main__':
    from model.train import DATA_DIR

    test_set = DATA_DIR.joinpath('rps-test-set', 'rps-test-set')
    test_images = {
        'paper': test_set.joinpath('paper', 'testpaper01-00.png'),
        'rock': test_set.joinpath('rock', 'testrock01-00.png'),
        'scissors': test_set.joinpath('scissors', 'testscissors01-00.png'),
    }
    model = load_model()
    for gt, image in test_images.items():
        batch = image_to_batch(image)
        pred, pred_pct = infer(model, batch)
        print(f'Prediction   :  {pred} ({pred_pct:.2%})')
        print(f'Ground truth :  {gt}')
        print('')
