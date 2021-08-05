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


def gen_3d_object_from_numpy(scan_array, slice_thickness, space_between_slices):
    """
    :param scan_array: scan slices from one patient with shape (~20, 128, 128)
    :param slice_thickness: dicom attribute SliceThickness (needs to be extracted from file of patient)
    :param space_between_slices: dicom attribute SpacingBetweenSlices (needs to be extracted from file of patient)
    :return: array of shape (z, 128, 128) where z is the height of the scanned region in mm
    """
    if len(scan_array.shape) == 4:
        ar = scan_array[:, :, :, 0]
    elif len(scan_array.shape) == 3:
        ar = scan_array
    else:
        sys.exit("wrong input shape for reconstruction. Needs to be of shape (z,x,y,channel) or (z,x,y)")
    out = np.zeros((1, ar.shape[1], ar.shape[2]))
    missing_per_slice = space_between_slices - slice_thickness
    spacing_missing_sum = 0
    for idx, mask in enumerate(ar):
        expanded_mask = np.expand_dims(mask, 0)
        if spacing_missing_sum > 1:
            array_to_add = np.repeat(expanded_mask, slice_thickness + 1, axis=0)
            spacing_missing_sum -= 1
        else:
            array_to_add = np.repeat(expanded_mask, slice_thickness, axis=0)
            spacing_missing_sum += missing_per_slice
        out = np.concatenate((out, array_to_add), axis=0)
    if spacing_missing_sum > 0.5:
        out = np.concatenate((out, expanded_mask), axis=0)
    out = np.moveaxis(out, 0, -1)
    return out


def calculate_volume(lung, x_spacing, y_spacing):
    """
    :param lung: array representing lung object of shape (z,x,y)
    :param x_spacing: first entry of the dicom attribute PixelSpacing, representing the width of one voxel
    :param y_spacing: second entry of the dicom attribute PixelSpacing, representing the depth of one voxel
    :return: volume estimation calculated via N_voxels * Volume of 1 Voxel
    """
    voxel_volume = x_spacing * y_spacing
    lung_volume = voxel_volume * np.sum(lung)
    return np.round(lung_volume / (10 ** 3), 4)


def array_to_coords(array, x_spacing, y_spacing):
    """
    :param array: array representing lung object of shape (z,x,y)
    :param x_spacing: first entry of the dicom attribute PixelSpacing, representing the width of one voxel
    :param y_spacing: second entry of the dicom attribute PixelSpacing, representing the depth of one voxel
    :return: three arrays of shapes (N_voxels, 1) containing x, y, z coordinates
    """
    x, y, z = array.nonzero()
    x = np.around(x * x_spacing, 2)
    y = np.around(y * y_spacing, 2)
    return x, y, z


def coords_to_array(x, y, z, x_spacing, y_spacing):
    """
    :param x: array of x-coordinate values of shape (N_voxels, 1)
    :param y: array of y-coordinate values of shape (N_voxels, 1)
    :param z: array of z-coordinate values of shape (N_voxels, 1)
    :param x_spacing: first entry of the dicom attribute PixelSpacing, representing the width of one voxel
    :param y_spacing: second entry of the dicom attribute PixelSpacing, representing the depth of one voxel
    :return: transform coordinate representation of lung object into array representation. return array of shape (z,x,y)
    """
    x = np.around(x / x_spacing)
    y = np.around(y / y_spacing)
    array = np.zeros((int(round(np.max(x) + 1)), int(round(np.max(y) + 1)),
                      int(round(np.max(z) + 1))))
    for i in range(len(x)):
        array[int(round(x[i])), int(round(y[i])), int(round(z[i]))] = 1
    return array


def kmeans_lung_split(x, y, z, z_scaling=1, return_coef_and_intercept=False):
    """
    :param x: array of x-coordinate values of shape (N_voxels, 1)
    :param y: array of y-coordinate values of shape (N_voxels, 1)
    :param z: array of z-coordinate values of shape (N_voxels, 1)
    :param z_scaling: factor with which the z-axis is scaled before applying K-Means algorithm with K=2, standard 1
    (no scaling)
    :param return_coef_and_intercept: return the coefficients and intercept of the linear SVM which is fitted after
    the KMeans
    :return: K-Means with K=2 is fitted on the coordinate data of the voxels two identify the two lungs.
    Outliers are then corrected by iteratively fitting a linear SVM until silhouette score does not improve
    """
    warnings.filterwarnings("ignore", message="Explicit.*")
    data = scale(np.array((x, y, z)).T)
    data[:, 2] = data[:, 2] * z_scaling
    k_means = KMeans(2, init='k-means++')
    k_means.fit(data)
    k_means_labels = k_means.predict(data).astype(bool)
    score = silhouette_score(data, k_means_labels, sample_size=5000)
    # iteratively fit svm until score does not improve
    svm = LinearSVC(C=0.0001)
    # revert z-scaling for svm's
    data[:, 2] = data[:, 2] * 1 / z_scaling
    data = scale(data)
    svm.fit(data, k_means_labels)
    old_labels = svm.predict(data)
    old_score = silhouette_score(data, old_labels, sample_size=5000)
    new_score = old_score + 0.0001
    i = 0
    while new_score > old_score:
        old_score = new_score
        svm.fit(data, old_labels)
        new_labels = svm.predict(data)
        new_score = silhouette_score(data, new_labels, sample_size=5000)
        old_labels = new_labels
    warnings.resetwarnings()
    if return_coef_and_intercept:
        return new_labels, new_score, svm.coef_, svm.intercept_
    else:
        return new_labels, new_score


def calculate_left_and_right_volume(object_3d, x_spacing, y_spacing):
    """
    :param object_3d: np-array of the shape (z,x,y) representing the voxels of the lung with 1-entries, empty space with
    0-entries
    :param x_spacing: first entry of the dicom attribute PixelSpacing, representing the width of one voxel
    :param y_spacing: second entry of the dicom attribute PixelSpacing, representing the depth of one voxel
    :return: the lung voxels are classified into two lungs. The volume of both lungs are calculated and the ratio of
    (greater volume/smaller volume) is returned. Should be an approximation for quantifying morphological differences
    between the two lungs
    """
    x, y, z = array_to_coords(object_3d, x_spacing, y_spacing)
    split_labels, silhouette = kmeans_lung_split(x, y, z)
    # figure out if x and y axes are switched in scan. Assumption: axis with greater diff between split_labels and
    # ~split_labels is x-axis
    if np.abs(np.mean(x[split_labels])-np.mean(x[~split_labels])) > \
            np.abs(np.mean(y[split_labels])-np.mean(y[~split_labels])):
        if np.mean(x[split_labels]) > np.mean(x[~split_labels]):
            right_lung = coords_to_array(x[split_labels], y[split_labels], z[split_labels], x_spacing, y_spacing)
            left_lung = coords_to_array(x[~split_labels], y[~split_labels], z[~split_labels], x_spacing, y_spacing)
        else:
            left_lung = coords_to_array(x[split_labels], y[split_labels], z[split_labels], x_spacing, y_spacing)
            right_lung = coords_to_array(x[~split_labels], y[~split_labels], z[~split_labels], x_spacing, y_spacing)
    else:
        if np.mean(y[split_labels]) > np.mean(y[~split_labels]):
            right_lung = coords_to_array(x[split_labels], y[split_labels], z[split_labels], x_spacing, y_spacing)
            left_lung = coords_to_array(x[~split_labels], y[~split_labels], z[~split_labels], x_spacing, y_spacing)
        else:
            left_lung = coords_to_array(x[split_labels], y[split_labels], z[split_labels], x_spacing, y_spacing)
            right_lung = coords_to_array(x[~split_labels], y[~split_labels], z[~split_labels], x_spacing, y_spacing)
    left_volume = calculate_volume(left_lung, x_spacing, y_spacing)
    right_volume = calculate_volume(right_lung, x_spacing, y_spacing)
    return left_volume, right_volume, round(silhouette, 4)


def extract_spacing_info(dicom_path, patient):
    """
    :param dicom_path: path where the input images are saved in the format patient_id.image_id.dcm
    :param patient: patient whose scan the spacing attributes should be extracted from
    :return: the dicom attributes PixelSpacing, SliceThickness and SpacingBetweenSlices which are essential for
    reconstructing a 3-dimensional object from the predictions
    """
    image_names = np.array(sorted(os.listdir(dicom_path)))
    patient_indices = np.array([patient in i for i in image_names])
    dcm = dcmread(os.path.join(dicom_path,image_names[patient_indices][0]))
    return dcm.PixelSpacing[0], dcm.PixelSpacing[1], dcm.SliceThickness, dcm.SpacingBetweenSlices
