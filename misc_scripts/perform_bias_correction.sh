
# ciss
src=/path/data_for_inference
src_segm=/pathdata_for_inference/output_from_nnunet_inference

# output files
tgt=/data/pulkit/exvivo_bias_correction/output_bias_corrected
interm_files=/data/pulkit/exvivo_bias_correction/intermediate_files

for subj in $(ls $src)
do
    subj=${subj: :-12}
    echo ${subj}

    c3d ${src_segm}/${subj}.nii.gz -thresh 1 inf 1 0 -dilate 1 2x2x2vox -holefill 1 -o ${interm_files}/${subj}_mask.nii.gz

    N4BiasFieldCorrection -d 3 \
        -i ${src}/${subj}_0000.nii.gz \
        -x ${interm_files}/${subj}_mask.nii.gz -t 0.3 0.01 200 \
        -o [${tgt}/${subj}_bias_corrected.nii.gz, ${tgt}/${subj}_bias_field.nii.gz]

    c3d ${tgt}/${subj}_bias_corrected.nii.gz -stretch 0.1% 99.9% 0 1000 -clip 0 1000 -o ${tgt}/${subj}_bias_corrected_norm.nii.gz

done
