# Welcome to hippogang's ex vivo world!
## Here, we will quickly run a docker container and get going with parcellating ex vivo 7 tesla T2w & flash MRI of human brain hemisphere. We will use our favourite nnU-Net!

#### Author: Pulkit Khandelwal
##### Version bare-bones: `docker_hippogang_exvivo_segm:v1.1.0`

##### Change Logs
08/30/24:
- Version `docker_hippogang_exvivo_segm:v1.4.0` updated with the model which includes the MTL, ventricles and the corpus callosum.

07/16/24:
- Version `docker_hippogang_exvivo_segm:v1.3.1` make singularity compatible by removing a copy command in the bash script

10/22/22:
- Version `docker_hippogang_exvivo_segm:v1.3.0` sgements even the white matter automatically now!
- Version `docker_hippogang_exvivo_segm:v1.2.0` now also performs segmentation for WMH in `in vivo` flair images used for Detre/Sandy's project on ADNI data.
- You can now also see a `logs.txt` file in the folder where you run the docker container from.

10/19/22:
- First created!

# Some useful things
- You don't need any working knowledge of docker or kubernetes, but if curious, here is a great [YouTube video](https://youtu.be/3c-iBn73dDE).
- You just need to provide a `nifti` image in the correct file format ending with `_0000.nii.gz`
- NO need for a GPU! Any linux-based machine works.
- Choose one of the following options which 
```
${OPTION}=exvivo_t2w
${OPTION}=exvivo_flash_more_subcort
${OPTION}=exvivo_ciss_t2w
${OPTION}=invivo_flair_wmh
```


# Sample data
I provide a sample `ex vivo` T2w image at the [box](https://upenn.box.com/s/q24zo6enivytnerko2ovt5kfzqq141ec) link. Use this image to test this docker container. There is also an `in vivo` flair image.

# Docker image
My docker image is located at `https://hub.docker.com/r/pulks/docker_hippogang_exvivo_segm`

# Docker files
I have provided some files in the folder `docker_files` for your reference only but we do not need those for running the demo.

# Steps
#### Step 0: Login to your lambda machine and open a new tmux session
`tmux new -s docker_trial`

#### Step 1: Prepare the data
Download the image from the box into a folder named `data_for_inference` (do NOT give it any other name) and then place this folder any directory of choice, for example, `/data/username/`.

#### Step 2: Pull the docker image
This should pull my docker image from docker hub. It is around 8GB in size.
`docker pull pulks/docker_hippogang_exvivo_segm:v${LATEST_TAG}`

#### Step 3: Run the docker container
Run the following command to start the inference. See how the volume is mounted in the following command. We mount the volume where the folder `data_for_inference`, with the image on which to run inference, is located. Here, `data_for_inference` is located in `/data/username/`. Leave the rest of the command as is.

`docker run --gpus all --privileged -v /data/username/:/data/exvivo/ -it pulks/docker_hippogang_exvivo_segm:v${LATEST_TAG} /bin/bash -c "bash /src/commands_nnunet_inference.sh ${OPTION} " >> logs.txt`

#### Voila! check the output!
It takes around 15-20 minutes to run the inference for the `ex vivo` T2w image. You should see a folder in your local machine at the path:
`/your/path/to/data_for_inference/output_from_nnunet_inference`

## White matter hypeintensities in `in vivo` FLAIR images
If, you want to run the WMH for `in vivo` flair data then run the following command. Make sure that the image is skull-stripped and normalized/standardized.
It takes around 1 minute to get the WMH segmentations in the `in vivo` FALIR image.
