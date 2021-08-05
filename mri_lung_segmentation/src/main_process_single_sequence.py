""" usage: main_process_single_sequence.py [-h] [-i imagesDir] [-m modelDir] [-s size] [-sp save_path] [-c cutOff] [-p plot]

Perform Lung Segmentations and Calculate Lung Volume for One MRI Sequence

arguments:
  -h, --help            show this help message and exit.
  -i imagesDir, --imagesDir
                        Path to the folder with MRI sequence images. Required argument.
  -m modelDir, --modelDir
                        Path to the folder with the pre-trained model weights.
  -s size, --size
                        Size (value,value) of the image for predictions.
  -t cutOff, --cutOff
                        Threshold value for predictions.
  -sp save, --save_path
                        Path to save the numpy masks for lung prediction
  -p plot, --plot       Set this flag if you want to plot the output.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pydicom import dcmread
from matplotlib import pyplot as plt
from DL_utils.model2D import unet2D

import segmentation_utils
import lung_volume_utils


def segmentation_volume_pipeline(input_sequence_path, models_path, size, cut_off, save_masks_path, generate_plot):
    # Load DICOM MRI Sequence
    patient_image_array = []
    sequence_names = os.listdir(input_sequence_path)
    sequence_names.sort()
    for i in sequence_names:
        if i[-4:] == '.dcm':
            dicom_sequence = dcmread(input_sequence_path+i)
            image_array = dicom_sequence.pixel_array
            # add channels dim
            image_array_tf = tf.expand_dims(image_array, axis=-1)
            patient_image_array.append(image_array_tf)

    # resize to target
    patient_image_array = np.array(patient_image_array)

    # Create Empty model
    segmentation_models = [unet2D(None, (size, size, 1), 1, "binary_crossentropy")]*3

    # Load Model Weights
    all_model_paths = []
    for file in os.listdir(models_path):
        if file.endswith(".hdf5"):
            all_model_paths.append(os.path.join(models_path, file))

    for idx, model in enumerate(segmentation_models):
        model.load_weights(all_model_paths[idx])

    # Standardize Sequence
    standardized_images = segmentation_utils.standardize_array_per_img(patient_image_array)
    assert np.max(standardized_images) == 1
    assert np.min(standardized_images) == 0

    cropped_images = segmentation_utils.crop_patient_images(
        standardized_images[:, :, :, 0],
        target_shape=size,
    )

    cropped_images_tf = tf.expand_dims(cropped_images, axis=-1)

    prediction = segmentation_utils.gen_maj_pred_of_images(segmentation_models, cropped_images_tf, cut_off)
    # Plot predicted masks
    if generate_plot:
        for i in range(cropped_images.shape[0]):
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(cropped_images[i, :, :], cmap='bone', aspect='auto')
            plt.subplot(122)
            plt.imshow(prediction[i, :, :, 0], cmap='bone', aspect='auto')
            plt.show()

    # Save predictions:
    with open(save_masks_path+'patient_predictions.npy', 'wb') as f:
        np.save(f, prediction)

    # Load DICOM Metadata
    x_spacing = dicom_sequence.PixelSpacing[0]
    y_spacing = dicom_sequence.PixelSpacing[1]
    slice_thickness = dicom_sequence.SliceThickness
    spacing_between_slices = dicom_sequence.SpacingBetweenSlices
    # Calculate volume
    object_3d = lung_volume_utils.gen_3d_object_from_numpy(prediction, slice_thickness, spacing_between_slices)
    left_volume, right_volume, silhouette = lung_volume_utils.calculate_left_and_right_volume(object_3d, x_spacing, y_spacing)
    overall_volume = lung_volume_utils.calculate_volume(object_3d, x_spacing, y_spacing)

    print('left_volume', left_volume)
    print('right_volume', right_volume)
    print('overall_volume', overall_volume)


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Perform Segmentations and Calculate Volume for One MRI Sequence",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imagesDir',
        help='Path to the folder with MRI sequence images',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-m', '--modelDir',
        help='Path to the folder with the pre-trained model weights.',
        default=None,
        required=True,
    )
    parser.add_argument(
        '-s', '--size',
        help='Image size, taken as (value,value) pixels for predictions',
        default=128,
        type=int)
    parser.add_argument(
        '-c', '--cutOff',
        help='Threshold value for predictions',
        default=0.5,
        type=int)
    parser.add_argument(
        '-sp', '--save_path',
        help='Path to save the generated masks',
        default=None,
        required=True,
    )
    parser.add_argument(
        '-p', '--plot',
        help='Set this flag if you want to plot the output.',
        action='store_true'
    )
    args = parser.parse_args()

    # Segment the lung and calculate the volumes
    segmentation_volume_pipeline(
        input_sequence_path=args.imagesDir,
        models_path=args.modelDir,
        size=args.size,
        cut_off=args.cutOff,
        save_masks_path=args.save_path,
        generate_plot=args.plot,
    )


if __name__ == '__main__':
    RUN_WITH_CMD = False  # Set to true if running from command line
    if RUN_WITH_CMD:
        main()
    else:
        # Segment the lung and calculate the volumes for One MRI Sequence
        segmentation_volume_pipeline(
            input_sequence_path='./mri_lung_segmentation/data/sequence/',
            models_path='./mri_lung_segmentation/models/all_pat',
            save_masks_path='./mri_lung_segmentation/data/predictions/',
            size=128,
            cut_off=0.5,
            generate_plot=True,
        )

# Usage example with args:
# python feature_extraction_and_regression/main_process_single_sequence.py -i 'data/BPD/sample/' -m 'models/all_pat' -sp 'results/BPD/sample/'
