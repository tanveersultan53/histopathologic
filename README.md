'''
# Histopathologic Cancer Detection with CNN

This project focuses on identifying metastatic cancer in histopathologic scans of lymph node sections using a Convolutional Neural Network (CNN). The dataset is derived from the Camelyon16 Challenge, hosted on Kaggle.

Achieved 95.1% AUC on the validation set using a custom CNN model with data augmentation and performance tracking.

## Dataset

- Source: [https://www.kaggle.com/c/histopathologic-cancer-detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data)
- ~220,000 96x96 .tif image patches
- Binary labels: 1 = cancer, 0 = normal
- Labeled CSV with image IDs and labels

## Project Structure

```
├── cancer_detection_cnn.py        # Core training script
├── dataset_utils.py               # PyTorch Dataset + transforms
├── best_cnn_model.pth             # Saved best model
├── README.md
└── images/
    └── training_plots.png         # Accuracy/AUC/Loss curves

```

## Model Architecture

- Input: 3x96x96
- Conv2D (32) → BN → ReLU → MaxPool
- Conv2D (64) → BN → ReLU → MaxPool
- Flatten → FC(600) → ReLU → Dropout
- FC(1) → Sigmoid

## Training Configuration

- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam (lr = 0.001)
- Batch Size: 32
- Epochs: 5
- Data Augmentation: Random crop, flip, rotation
- Normalization: ImageNet mean/std

## Results

Epoch | Val AUC | Val Accuracy | Train Loss
------|---------|--------------|-----------
1     | 0.9037  | 82.10%       | 0.4656
3     | 0.9388  | 85.02%       | 0.3739
5     | 0.9511  | 87.79%       | 0.3252

## Future Work

- Add Grad-CAM explainability
- Try ResNet or Vision Transformers
- Apply Weak Supervision or MIL
- Deploy to web via MediSight-style UI

## Author

Muhammad Tanveer Sultan  
MSc Advanced Data Science | AI in Healthcare  
Email: engr.tanveersultan53@gmail.com  
Demo: https://medisight.netlify.app

## License

MIT License
'''
