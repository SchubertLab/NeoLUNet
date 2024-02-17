# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join("..", "src"))
import pandas as pd
pd.set_option('display.max_columns', 50)
from functools import reduce
import lung_volume_utils
import segmentation_utils
import seaborn as sns
from DL_utils.model2D import unet2D
from datetime import date
from tqdm import tqdm

# needed helper functions
def get_masks_with_names_indices(data_set, leave_out):
    groundtruth_path = f'../data/{data_set}/train/groundtruth/'
    image_names = sorted(os.listdir(groundtruth_path))
    if "giessen" in data_set:
        patient_names = [image_name[:6] for image_name in image_names if image_name[:6] == leave_out]
        image_indices = [image_name[7:9] for image_name in image_names if image_name[:6] == leave_out]
        test_masks = np.array([np.load(f'{groundtruth_path}{mask}') for mask in image_names if mask[:6] == leave_out])
    else:
        patient_names = [image_name[:7] for image_name in image_names if image_name[:7] == leave_out]
        image_indices = [image_name[8:10] for image_name in image_names if image_name[:7] == leave_out]
        test_masks = np.array([np.load(f'{groundtruth_path}{mask}') for mask in image_names if mask[:7] == leave_out])
    return test_masks, patient_names, image_indices

def get_all_images_with_names_indices(data_set):
    image_path = f'../data/{data_set}/train/image/'
    image_names = sorted(os.listdir(image_path))
    if "giessen" in data_set:
        patient_names = [image_name[:6] for image_name in image_names]
        image_indices = [image_name[7:9] for image_name in image_names]
        test_images = np.array([np.load(f'{image_path}{mask}') for mask in image_names])
    else:
        patient_names = [image_name[:7] for image_name in image_names]
        image_indices = [image_name[8:10] for image_name in image_names]
        test_images = np.array([np.load(f'{image_path}{mask}') for mask in image_names])
    return test_images, patient_names, image_indices

def dice_loss_per_slice(masks, predictions):
    dice_loss_list = []
    for idx, mask in enumerate(masks):
        dice_loss = dice_loss_keras(mask, predictions[idx])
        dice_loss_list.append(dice_loss)
    return np.array(dice_loss_list)

def dice_loss_per_patient(masks, predictions, patient_names):
    patient_dice_losses = np.ones(len(predictions))
    for patient in list(set(patient_names)):
        patient_indices = np.array([p == patient for p in patient_names])
        dice_loss = dice_loss_keras(masks[patient_indices], predictions[patient_indices])
        patient_dice_losses[patient_indices] = dice_loss
    return patient_dice_losses
    
def dice_loss_keras(y_true, y_pred):
    smooth = 1.
    image_f = y_true.flatten()
    prediction_f = y_pred.flatten()
    intersection = image_f * prediction_f
    score = (2. * np.sum(intersection) + smooth) / (np.sum(np.square(image_f),-1) + np.sum(np.square(prediction_f),-1) + smooth)
    return 1. - score
    
def evaluation_main(cohorts, image_qualities_and_sev, lopo_model_path, all_pat_model_path, dicom_path, empty_models, cut_off):
    # initialization for evaluation
    file_not_found_count = 0
    gt_not_found_list = []
    overall_results = {}
    for cohort in cohorts:
        if cohort == 'Großhadern':
            image_data_set = 'P1'
            dicom_path = os.path.join('..', 'data','AIRR_Images_no_folders')
        elif cohort == 'Gießen':
            image_data_set = 'giessen_P1'
            dicom_path = os.path.join('..', 'data','AIRR_Images_giessen_no_folders')
        else:
            print('Invalid cohort')
            raise
        input_images, patient_names, image_indices = get_all_images_with_names_indices(image_data_set)
        distinct_patient_names = list(set(patient_names))
        leave_out_df_cohort = pd.DataFrame()
        print(f'start evaluation for cohort {cohort}')
        for leave_out in tqdm(distinct_patient_names):
            # load models
            patient_indices = np.array(patient_names) == leave_out
            input_images_for_patient = input_images[patient_indices]
            models, lopo_model_used = segmentation_utils.load_weights_for_patient(empty_models, lopo_model_path,
                                                                           all_pat_model_path, leave_out)
            standardized_images = segmentation_utils.standardize(input_images_for_patient)
            assert np.max(standardized_images) == 1
            assert np.min(standardized_images) == 0
            prediction_list = segmentation_utils.gen_prediction_list_of_images(models, standardized_images, cut_off)
            pred_p1 = prediction_list[0][:,:,:,0]
            pred_p2 = prediction_list[1][:,:,:,0]
            pred_p3 = prediction_list[2][:,:,:,0]
            
            if cohort=="Großhadern":
                ds_names = ['P1', 'P2', 'P3']
            else:
                ds_names = ['giessen_P1', 'giessen_P2', 'giessen_P3']

            test_masks_p1, patient_names_p1, image_indices_p1 = get_masks_with_names_indices(ds_names[0], leave_out) 
            test_masks_p2, patient_names_p2, image_indices_p2 = get_masks_with_names_indices(ds_names[1], leave_out)
            test_masks_p3, patient_names_p3, image_indices_p3 = get_masks_with_names_indices(ds_names[2], leave_out)
            #assert (patient_names_p1==patient_names_p2 and patient_names_p1==patient_names_p3)
            #assert (image_indices_p1==image_indices_p2 and image_indices_p1==image_indices_p3)
            if test_masks_p1.shape[0]==0:
                gt_not_found_list.append([leave_out, test_masks_p1.shape, test_masks_p2.shape, test_masks_p3.shape])
                print(f'did not find patient {leave_out}')
                continue
            if not (patient_names_p1==patient_names_p2 and patient_names_p1==patient_names_p3):
                continue
            test_masks_majority = ((test_masks_p1+test_masks_p2+test_masks_p3)>=2).astype(int)
            pred_majority = (pred_p1+pred_p2+pred_p3>=2).astype(int)
            k = len(pred_p1)

            # build list of image qualities and severities for patient
            qualities = []
            severities = []
            for pat in patient_names_p1:
                quality = list(image_qualities_and_sev[image_qualities_and_sev['id']==pat]["scan_quality"])
                sev = list(image_qualities_and_sev[image_qualities_and_sev['id']==pat]["bpd_severity"])
                qualities.append(quality[0])
                severities.append(sev[0])

            # mean dice score single annotator vs other annotators and single model vs other annotators
            p1_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p2, test_masks_p1) + dice_loss_per_slice(test_masks_p3, test_masks_p1))
            p2_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p1, test_masks_p2) + dice_loss_per_slice(test_masks_p3, test_masks_p2))
            p3_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p1, test_masks_p3) + dice_loss_per_slice(test_masks_p2, test_masks_p3))
            model_p1_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p2, pred_p1) + dice_loss_per_slice(test_masks_p3, pred_p1))
            model_p2_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p1, pred_p2) + dice_loss_per_slice(test_masks_p3, pred_p2))
            model_p3_vs_rest = 0.5 * (dice_loss_per_slice(test_masks_p1, pred_p3) + dice_loss_per_slice(test_masks_p2, pred_p3))
            # mean patient dice score single annotator vs other annotators and single model vs other annotators
            p1_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p2, test_masks_p1, patient_names_p1) +\
                                         dice_loss_per_patient(test_masks_p3, test_masks_p1, patient_names_p1))
            p2_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p1, test_masks_p2, patient_names_p1) +\
                                         dice_loss_per_patient(test_masks_p3, test_masks_p2, patient_names_p1))
            p3_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p1, test_masks_p3, patient_names_p1) +\
                                         dice_loss_per_patient(test_masks_p2, test_masks_p3, patient_names_p1))
            model_p1_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p2, pred_p1, patient_names_p1) +\
                                               dice_loss_per_patient(test_masks_p3, pred_p1, patient_names_p1))
            model_p2_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p1, pred_p2, patient_names_p1) +\
                                               dice_loss_per_patient(test_masks_p3, pred_p2, patient_names_p1))
            model_p3_vs_rest_patient = 0.5 * (dice_loss_per_patient(test_masks_p1, pred_p3, patient_names_p1) +\
                                               dice_loss_per_patient(test_masks_p2, pred_p3, patient_names_p1))
            
            # calculate volumes based on the prediction of single raters
            # calculate for P1s masks
            x_spacing, y_spacing, slice_thickness, spacing_between_slices = lung_volume_utils.extract_spacing_info(dicom_path, leave_out)
            gt_object_3d = lung_volume_utils.gen_3d_object_from_numpy(test_masks_p1, slice_thickness, spacing_between_slices)
            pred_object_3d = lung_volume_utils.gen_3d_object_from_numpy(pred_p1, slice_thickness, spacing_between_slices)
            gt_volume_p1 = lung_volume_utils.calculate_volume(gt_object_3d, x_spacing, y_spacing)
            pred_volume_p1 = lung_volume_utils.calculate_volume(pred_object_3d, x_spacing, y_spacing)
            #calculate for P2s masks
            gt_object_3d = lung_volume_utils.gen_3d_object_from_numpy(test_masks_p2, slice_thickness, spacing_between_slices)
            pred_object_3d = lung_volume_utils.gen_3d_object_from_numpy(pred_p2, slice_thickness, spacing_between_slices)
            gt_volume_p2 = lung_volume_utils.calculate_volume(gt_object_3d, x_spacing, y_spacing)
            pred_volume_p2 = lung_volume_utils.calculate_volume(pred_object_3d, x_spacing, y_spacing)
            # calculate for P3s mask
            gt_object_3d = lung_volume_utils.gen_3d_object_from_numpy(test_masks_p3, slice_thickness, spacing_between_slices)
            pred_object_3d = lung_volume_utils.gen_3d_object_from_numpy(pred_p3, slice_thickness, spacing_between_slices)
            gt_volume_p3 = lung_volume_utils.calculate_volume(gt_object_3d, x_spacing, y_spacing)
            pred_volume_p3 = lung_volume_utils.calculate_volume(pred_object_3d, x_spacing, y_spacing)

            # last ones: calculate "majority vs. majority" scores...
            maj_on_maj = dice_loss_per_slice(test_masks_majority, pred_majority)
            maj_on_maj_patient = dice_loss_per_patient(test_masks_majority, pred_majority, patient_names_p1)
            # ... and volume
            gt_maj_object_3d = lung_volume_utils.gen_3d_object_from_numpy(test_masks_majority, slice_thickness, spacing_between_slices)
            pred_maj_object_3d = lung_volume_utils.gen_3d_object_from_numpy(pred_majority, slice_thickness, spacing_between_slices)
            gt_maj_volume = lung_volume_utils.calculate_volume(gt_maj_object_3d, x_spacing, y_spacing)
            pred_maj_volume = lung_volume_utils.calculate_volume(pred_maj_object_3d, x_spacing, y_spacing)
            left_volume, right_volume, _ = lung_volume_utils.calculate_left_and_right_volume(pred_maj_object_3d, x_spacing, y_spacing)
            pred_maj_volume_ratio = max(left_volume/right_volume, right_volume/left_volume)
            


            # create result data frame
            leave_outs=[leave_out]*k; epochs = [300]*k; loss_functions = ['binary_crossentropy']*k;\
            learning_rates = [0.001]*k; augs = ['augs1']*k; pre_processing_fcts = ['None']*k;\
            thresholds = [cut_off]*k
            temp_df = pd.DataFrame({"patient": patient_names_p1, "image_quality":qualities, 'bpd_severity':severities, "leave_out":leave_outs,\
                                    "loss_fct":loss_functions, "learning_rate":learning_rates, "augs": augs,\
                                    "pre_processing_fct":pre_processing_fcts,"epochs":epochs, "image_id":image_indices_p1,\
                                    "p1_vs_rest":p1_vs_rest, "p2_vs_rest":p2_vs_rest, "p3_vs_rest":p3_vs_rest,\
                                    "model_p1_vs_rest":model_p1_vs_rest, "model_p2_vs_rest":model_p2_vs_rest,  "model_p3_vs_rest":model_p3_vs_rest,\
                                    "p1_vs_rest_patient":p1_vs_rest_patient, "p2_vs_rest_patient":p2_vs_rest_patient, "p3_vs_rest_patient":p3_vs_rest_patient,\
                                    "model_p1_vs_rest_patient":model_p1_vs_rest_patient, "model_p2_vs_rest_patient":model_p2_vs_rest_patient,\
                                    "model_p3_vs_rest_patient":model_p3_vs_rest_patient,\
                                    "gt_volume_p1": gt_volume_p1, "pred_volume_p1": pred_volume_p1,\
                                    "gt_volume_p2": gt_volume_p2, "pred_volume_p2": pred_volume_p2,\
                                    "gt_volume_p3": gt_volume_p3, "pred_volume_p3": pred_volume_p3,\
                                    "maj_on_maj": maj_on_maj, "maj_on_maj_patient": maj_on_maj_patient, "gt_maj_volume": gt_maj_volume,
                                    "pred_maj_volume": pred_maj_volume, 'pred_maj_volume_ratio': pred_maj_volume_ratio})
            leave_out_df_cohort = leave_out_df_cohort.append(temp_df, ignore_index=True)
            
            mask_list = [test_masks_p1, test_masks_p2, test_masks_p3, test_masks_majority]
            pred_list = [pred_p1, pred_p2, pred_p3, pred_majority]
            names = ['P1', 'P2', 'P3', 'Majority']
            slice_dice_loss_list = [p1_vs_rest, p2_vs_rest, p3_vs_rest, maj_on_maj]
        leave_out_df_cohort['cohort'] = cohort
        overall_results[cohort] = leave_out_df_cohort
    return overall_results, file_not_found_count, gt_not_found_list
    
    
# load quality rating and disease severities
image_qualities_and_sev = pd.read_csv('../data/quality_grading_and_severities.csv')

# configurations for evaluation
cohorts = ['Großhadern', 'Gießen']
lopo_model_path = os.path.join('..', 'final_models','lopo')
all_pat_model_path = os.path.join('..', 'final_models','all_pat')
dicom_path = os.path.join('..', 'data','AIRR_Images_no_folders')
empty_models = [unet2D(None, (128, 128, 1), 1, "binary_crossentropy")]*3
cut_off = 0.5

overall_results_dict, file_not_found_count, gt_not_found_list = evaluation_main(cohorts, image_qualities_and_sev, lopo_model_path, all_pat_model_path, dicom_path, empty_models, cut_off)
    
overall_df = pd.DataFrame()
for cohort in cohorts:
    overall_df = overall_df.append(overall_results_dict[cohort], ignore_index=True)
overall_df.to_csv(f"../data/all_seg_scores_per_image_{date.today()}.csv", index=False)