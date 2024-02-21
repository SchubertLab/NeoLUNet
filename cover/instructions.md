# NeoLUNet -  Instructions

## 1. Clone the repository
> ```bash
> git clone 'git@github.com:SchubertLab/NeoLUNet.git'
> ```
> - Install dependencies
> ```bash
> conda env create -f neolunet_env.yml

## 2. Input Data Preparation
> Input file format should be DICOM (.dcm).
> 
> Create a folder with the corresponding patient id in the following location:
> >"mri_lung_segmentation/data/{patient_id}"
> 
> Place the .dcm files of the patient MRI images in a sub folder called 'input_sequences'
> >"mri_lung_segmentation/data/{patient_id}/input_sequences/".
> 
> The files should be alphabetically ordered, i.e. (mri_1.dcm, mri_2.dcm...). 
>
> The first file should refer to the image with z=0.

## 3. Downloading Model Weights
> The models can be downloaded from the <a href=https://doi.org/10.5281/zenodo.10686751>Zenodo Link</a>
> 
> You should download 3 models (unet_p1.hdf5, unet_p2.hdf5, unet_p3.hdf5), corresponding to 3 different ground truth annotations.
> > Create the directory "mri_lung_segmentation/models/all_pat" and place the hdf5 files there.

## 4. Running the segmentation script:
> For running the segmenation script, move to "mri_lung_segmentation/src" in your terminal
> ```bash
> cd mri_lung_segmentation/src'
> ```
> Run the script by providing the only required input argument which is the location of your patient folder:
> ```bash
> python main_process_single_sequence.py -i '../data/{patient_id}'
> ```

## 5. Which results are generated?
> In your patient folder
> > "mri_lung_segmentation/data/{patient_id}" 
> 
> there are now 3 folders:
> - plots: a folder with png files visualizing the model predictions on each mri image.
> - extracted_features: containing a csv file with the extracted features.
> - predictions: (.npy) file containing the model predictions for all provided images, to be used for feature extraction.
