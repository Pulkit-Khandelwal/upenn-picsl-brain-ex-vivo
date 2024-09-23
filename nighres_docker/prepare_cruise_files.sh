src_dir=/data/cruise_files/data_for_topology_correction
tgt_dir=/data/cruise_files/data_for_topology_correction/output_corrected_topology

mkdir -p ${tgt_dir}
for filename in $(ls ${src_dir})
do
  filename_subj="${filename%???????}"
  echo $filename $filename_subj

  ./data/c3d ${src_dir}/${filename} -replace 2 7 3 7 4 7 5 7 6 7 -o ${tgt_dir}/${filename_subj}_gm_wm.nii.gz
  ./data/c3d ${tgt_dir}/${filename_subj}_gm_wm.nii.gz -replace 7 0 -o ${tgt_dir}/${filename_subj}_gm.nii.gz
  ./data/c3d ${tgt_dir}/${filename_subj}_gm_wm.nii.gz -replace 1 0 7 1 -o ${tgt_dir}/${filename_subj}_wm.nii.gz
  ./data/c3d ${tgt_dir}/${filename_subj}_gm_wm.nii.gz -replace 1 0 7 0 0 1 -o ${tgt_dir}/${filename_subj}_csf.nii.gz

done;

python3 /data/nighres_topology_corection.py