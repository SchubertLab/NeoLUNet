import os
import sys
import warnings
import pandas as pd
import numpy as np
from DL_utils.model2D import unet2D
from pydicom import dcmread
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from datetime import date

import segmentation_utils
import lung_volume_utils


def get_images_and_patients_from_path(dicom_path, return_original_shapes=False):
    """
    :param dicom_path: path where the input images are saved in the format patient_id.image_id.dcm
    :param return_original_shape: flag to determine if original shape should be returned
    :return: an array of shape (n_patients, n_slices_of_patient, x, y) containing all patients' images, a list of same
    length containing the patient IDs
    """
    # Don't use this in evaluation of Segmentation (cuts need to match cuts from masks)!
    image_names = np.array(sorted(os.listdir(dicom_path)))
    patient_names = np.array([fn.split('.')[0] for fn in image_names])
    distinct_patients = sorted(list(set(patient_names)))
    patient_arrays = []
    for p in distinct_patients:
        patient_indices = patient_names == p
        patient_files = image_names[patient_indices]
        patient_array = np.array([dcmread(dicom_path + f).pixel_array for f in patient_files])
        original_shapes = [array.shape for array in patient_array]
        patient_arrays.append(patient_array)
    patient_arrays = np.array(patient_arrays)
    if return_original_shapes:
        return patient_arrays, distinct_patients, original_shapes
    else:
        return patient_arrays, distinct_patients
    

def get_masks_and_patients_from_path(dicom_path, return_original_shapes=False):
    """
    :param dicom_path: path where the input images are saved in the format patient_id.image_id.dcm
    :param return_original_shape: flag to determine if original shape should be returned
    :return: an array of shape (n_patients, n_slices_of_patient, x, y) containing all patients' images, a list of same
    length containing the patient IDs
    """
    image_names = np.array(sorted(os.listdir(dicom_path)))
    patient_names = np.array([fn.split('.')[0] for fn in image_names])
    distinct_patients = sorted(list(set(patient_names)))
    cropped_arrays = []
    for p in distinct_patients:
        patient_indices = patient_names == p
        patient_files = image_names[patient_indices]
        patient_array = np.array([dcmread(dicom_path + f).pixel_array for f in patient_files])
        original_shapes = [array.shape for array in patient_array]
        cropped_array = segmentation_utils.crop_patient_images(patient_array)
        cropped_arrays.append(cropped_array)
    cropped_arrays = np.array(cropped_arrays)
    if return_original_shapes:
        return cropped_arrays, distinct_patients, original_shapes
    else:
        return cropped_arrays, distinct_patients


def get_images_of_patient(patient, distinct_patient_names, input_images):
    """
    :param patient: ID of patient whose images should be returned
    :param distinct_patient_names: list of all patients. Needs the same shape as input_images
    :param input_images: array containing the scans of each patient.
    :return: one array of shape (~20, 128, 128) containing all images of the given patient
    """
    patient_indices = np.array(distinct_patient_names) == patient
    return input_images[patient_indices][0]


def reconstruction_main(dicom_path, lopo_model_path, all_pat_model_path, cut_off_value, gen_plots=False):
    """

    :param dicom_path: path where the input images are saved in the format patient_id.image_id.dcm
    :param lopo_model_path: path of models trained in leave-one-patient-out mode
    :param all_pat_model_path: path of models trained on all patients which is used for reconstruction on new data
    :param cut_off_value: threshold for binarization of the UNET's predictions
    :param gen_plots: if true, the 10th slice of each patient's input image will be plotted next to the prediction
    :return: data frame with volume estimations for each patient with images in the dicom-folder
    """
    input_images, distinct_patient_names, original_shapes = \
        get_images_and_patients_from_path(dicom_path, return_original_shapes=True)
    left_volumes = []
    right_volumes = []
    overall_volumes = []
    silhouette_scores = []
    lopo_model_used_list = []
    empty_models = [unet2D(None, (None, None, 1), 1, "binary_crossentropy")]*3
    for patient in distinct_patient_names:
        input_images_for_patient = get_images_of_patient(patient, distinct_patient_names, input_images)
        models, lopo_model_used = segmentation_utils.load_weights_for_patient(empty_models, lopo_model_path,
                                                                       all_pat_model_path, patient)
        standardized_images = segmentation_utils.standardize(input_images_for_patient)
        assert np.max(standardized_images) == 1
        assert np.min(standardized_images) == 0
        prediction = segmentation_utils.gen_maj_pred_of_images(models, standardized_images, cut_off_value)
        if gen_plots:
            plt.subplot(121)
            plt.imshow(standardized_images[10, :, :], cmap='bone', aspect='auto')
            plt.subplot(122)
            plt.imshow(prediction[10, :, :, 0], cmap='bone', aspect='auto')
            plt.show()
        x_spacing, y_spacing, slice_thickness, spacing_between_slices = lung_volume_utils.extract_spacing_info(dicom_path, patient)
        object_3d = lung_volume_utils.gen_3d_object_from_numpy(prediction, slice_thickness, spacing_between_slices)
        left_volume, right_volume, silhouette = lung_volume_utils.calculate_left_and_right_volume(object_3d, x_spacing, y_spacing)
        overall_volume = lung_volume_utils.calculate_volume(object_3d, x_spacing, y_spacing)
        left_volumes.append(left_volume)
        right_volumes.append(right_volume)
        overall_volumes.append(overall_volume)
        silhouette_scores.append(silhouette)
        lopo_model_used_list.append(lopo_model_used)
        print('-------------------------')
        print(f'reconstructed lung of patient {patient}')
        print(f'found {standardized_images.shape[0]} images for patient')
        print(f'lopo model used/found: {lopo_model_used}')
        print(f'spacing attributes: {x_spacing, y_spacing, slice_thickness, spacing_between_slices}')
        print(f'left, right and overall volume: {left_volume, right_volume, overall_volume}')
    result_df = pd.DataFrame({'patient': distinct_patient_names, 'left_volume': left_volumes,
                              'right_volume': right_volumes, 'volume': overall_volumes, 'silhouette': silhouette_scores,
                              'lopo_model_used': lopo_model_used_list})
    return result_df

### CONFIG #####

# path to trained model(s) from home directory
lopo_model_path = os.path.join('..', 'final_models','lopo')
all_pat_model_path = os.path.join('..', 'final_models','all_pat')
dicom_path = '../data/AIRR_Images_no_folders/'
# last char in dicom path / for volume data set name to work
assert dicom_path[-1] == '/'

cut_off_value = 0.5

gen_plots = False


volume_data_set_path = \
    f'extracted_features/reconstruction_from_{dicom_path.split("/")[-2]}' \
    f'_{date.today()}.csv'

print(volume_data_set_path)
volume_df = reconstruction_main(dicom_path, lopo_model_path, all_pat_model_path, cut_off_value, gen_plots=gen_plots)
volume_df.to_csv(volume_data_set_path, index=False)

