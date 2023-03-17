
echo "Spinning up virtual environment and setting up nnUNet......"
source /ml/bin/activate 

cp -r /src/network_architecture/* /ml/src/nnunet/nnunet/network_architecture/
cp -r /src/network_trainer/nnUNetTrainerWMHV2.py /ml/src/nnunet/nnunet/training/network_training/

export nnUNet_preprocessed="/src/nnunet_paths_that_it_requires"
export RESULTS_FOLDER="/src/nnunet_paths_that_it_requires"
export nnUNet_raw_data_base="/src/nnunet_paths_that_it_requires"

echo "Getting you pretty segmentations....."

nnUNet_predict -i /data/exvivo/data_for_inference/ -o /data/exvivo/data_for_inference/output_from_nnunet_inference -t 386 -tr nnUNetTrainerWMHV2 -m 3d_fullres --disable_mixed_precision -f all
