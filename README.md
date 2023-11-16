# Liver-Tumor_Segmentation
If not using Google Colab, use .py files for training and testing the model, currently UNett_ASPP is available. The original training dataset contains nii files should be placed as follow:

CT Scans: /content/drive/My Drive/EC523Project/VolumeData/*

Labels: /content/drive/My Drive/EC523Project/SegmentationData/segmentations

For Google Colab, copy the data as /content/drive/My Drive/EC523Project/Resize_augmented(512x512x512)/* with given pkl files, and place original dataset as described above

Models: UNett_ASPP and TransUNet

Dataset: LiTS17 https://www.kaggle.com/datasets/javariatahir123/lits17-liver-tumor-segmentation?select=CT_Mask, training: 100, validation: 30, testing: 70

Loss: Dice Loss
