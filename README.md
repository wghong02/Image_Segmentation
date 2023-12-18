# Liver-Tumor_Segmentation
If not using Google Colab, use .py files for training and testing the model, but .ipynb is recommended. The original training dataset contains .nii files should be placed as follow:

CT Scans: /content/drive/My Drive/EC523Project/VolumeData/*

Labels: /content/drive/My Drive/EC523Project/SegmentationData/segmentations

Dataset: LiTS17 https://www.kaggle.com/datasets/javariatahir123/lits17-liver-tumor-segmentation?select=CT_Mask, training: 100, validation: 30, testing: 70.

For Google Colab, copy the data as /content/drive/My Drive/EC523Project/Resize_augmented(512x512x512)/* with given pkl files, and place original dataset as described above.

UNet_ASPP: load model from unet_aspp_classweight124_30_dce3+ce7.pth

Model architecture: ![UNet_ASPP](/imgs/unet_aspp_model.png)

Result: ![UNet_ASPP Result](/imgs/unet_aspp_result.png)