# Rock-Paper-Scissors Game

A rock-paper-scissor game implementation using hand gestures inferred from
webcam input.

## Pipeline

Webcam input is implemented using `OpenCV` videocapture elements.

Hand gestures can be classified by:

- using hand landmarks detections based on `MediaPipe`
- custom trained CNN image classifier using `PyTorch`

The output in rendered as `OpenCV` overlay depicting the webcam image,
the selected region-of-interest to classify hand gestures, the selected
move of the opponent, i.e., the computer's random choice, and the current
scores.

## Usage

The controller takes the following arguments:

- camera input, i.e., index of camera
- predictor: Landmark detection or CNN
- ROI to extract images for the predictor within the webcam image

These can be set directly in `main.py`.

Afterward, the game can be started by running:

```bash
python -m main
```
