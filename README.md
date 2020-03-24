# DepthCompletionVISLAM

# Abstract

This paper addresses the problem of learning to complete a scene's depth from sparse depth points and images of indoor scenes.
Specifically, we study the case in which the sparse depth is computed from a visual-inertial simultaneous localization and mapping (VI-SLAM) system.
The resulting point cloud has low density, it is noisy, and has non-uniform spatial distribution, as compared to the conventional study of sparse input from active depth sensors, e.g., LiDAR or Kinect.
Since the VI-SLAM produces point clouds only over high-textured areas, we compensate the missing depth of the low textured ones by leveraging their planar structures and their surface normals which is an important intermediate representation.
The pre-trained surface normal network suffers from large performance degradation when there is a significant difference in the viewing direction of the test image as compared to the trained ones (especially the roll angle).
Therefore, we use the available gravity estimate from the VI-SLAM to warp the input image to the orientation prevailing in the training dataset.
This results in a significant performance gain for the surface normal estimate, and thus the dense depth estimates.
Figure 1 overviews our pipeline, ![Alt text](images/overview.jpg?raw=true).

# Installation Guide
For convenience, all the code in this repositority are assumed to be run inside NVIDIA-Docker.

### For instructions on installing NVIDIA-Docker, please follow the following steps (note that this is for Ubuntu 18.04):

For more detailed instructions, please refer to [this link](https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/).
1. Install Docker

    ```
    sudo apt-get update

    sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent \
        software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    sudo apt-key fingerprint 0EBFCD88

    sudo add-apt-repository \
       "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"

    sudo apt-get update

    sudo apt-get install docker-ce docker-ce-cli containerd.io
    ```

    To verify Docker installation, run:

    ```
    sudo docker run hello-world
    ```

2. Install NVIDIA-Docker

    ```
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
      sudo apt-key add -

    sudo apt-get install nvidia-docker2

    sudo pkill -SIGHUP dockerd
    ```

To activate the docker environment, run the following command:

```
nvidia-docker run -it --rm --ipc=host -v /:/home nvcr.io/nvidia/pytorch:19.08-py3
```

where `/` is the directory in the local machine (in this case, the root folder), and `/home` is the reflection of that directory in the docker.
This has also specified NVIDIA-Docker with PyTorch version 19.08 which is required to ensure the compatibility
between the packages used in the code (at the time of submission).

Inside the docker, change the working directory to this repository:
```
cd /home/PATH/TO/THIS/REPO/DepthCompletionVISLAM
```

# Quick Inference

Please follow the below steps to run sparse-to-dense depth completion using our provided pre-trained model:

1. Please download and extract the files provided by this [link](https://drive.google.com/a/umn.edu/file/d/1pXG-kFWOzXaRkrmGqMdD3OXZXBHqlUIY/view?usp=sharing) to [`./checkpoints/`](https://github.com/MARSLab-UMN/vi_depth_completion/blob/master/checkpoints) directory.

Make sure you have the following files inside [`./checkpoints/`](https://github.com/MARSLab-UMN/vi_depth_completion/blob/master/checkpoints) folder: `depth_completion.ckpt`, `surface_normal_fpn_use_gravity.ckpt`, `plane_detection.pth`.

2. To run on any dataset other than the demo mode, please run the corresponding dataset creator in the data directory (e.g., for [Azure Kinect dataset](https://drive.google.com/open?id=1O-WCNFna7DVUEst92K7Cln545FNWMz9U), run [data/dataset_creator_azure.py](https://github.com/MARSLab-UMN/vi_depth_completion/blob/master/data/dataset_creator_azure.py) to generate the dataset's .pkl file. The path to this file must be provided using the command line option --dataset_pickle_file when running main.py.

3. Run [`demo.sh`](https://github.com/MARSLab-UMN/vi_depth_completion/blob/master/demo.sh) to extract the results in [`./demo_dataset/`](https://github.com/MARSLab-UMN/vi_depth_completion/blob/master/demo_dataset). This file also contains sample commands to run other datasets.

Figure 2 shows the dense depth estimates together with intermediate outputs from the demo images.
![Alt text](images/sample_results.jpg?raw=true)
