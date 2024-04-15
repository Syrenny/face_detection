# Face Detection System for Security Checkpoints

This repository contains a face detection system designed for security checkpoints. The system utilizes a deep learning model to detect faces in images or video streams, aiding in the security screening process.

## Installation

To install the necessary dependencies for running the face detection system, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Model Weights

The `best.onnx` file contains the pre-trained weights of our neural network model for face detection. These weights are essential for running the detection algorithm.

## Integration

The `bot_integration.py` file contains the core class `Yolov8Detector()` responsible for face detection. This class provides methods for loading the model, performing inference on images, and drawing bounding boxes around detected faces.

## Usage Example

For a demonstration of how to use our face detection system, refer to the `example.ipynb` notebook. This notebook provides a step-by-step guide on how to load the model, process images, and visualize the detection results.

## License

This project is licensed under the MIT License, allowing for both personal and commercial use with proper attribution.

## Contributors

- [Ilia Syrenny](https://github.com/Syrenny)
- [Piotr Podstavkin](https://github.com/pritor)
- [Mikhail Makeev](https://github.com/kramdm)
- [Olga Petrochenko](https://github.com/odheL42)
- [Vyacheslav Chelondaev](https://github.com/VChelondaev)
