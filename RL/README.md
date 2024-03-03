# Soft Actor-Critic algorithm (Baseline)



Project Description: This is a repo that contains the implementation of the SAC (Soft Actor-Critic) algorithm to act as the baseline for the Unsupervised Pretraining algorithm

## Installing the Environment
There are two distinct ways to install the necessary packages and dependencies.

We recommend that the installation processes be tried in the same order that they appear in the current README.md file

On Vulcan:

```
module load anaconda/3/2019.03
module load cuda/11.4
```

Method 1:
Assuming there is access to a GPU that can run CUDA 9.2 (or higher version)

```shell
conda env create -f conda_env.yml
source activate pytorch_sac
```

Method 2:
create a new environment using conda and activate it 

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install absl-py
conda install pyparsing
pip install termcolor
pip install git+https://github.com/deepmind/dm_control.git
pip install git+https://github.com/denisyarats/dmc2gym.git
pip install tb-nightly
pip install pyyaml
```
After installing the above packages, open the conda environment

If on a headless LINUX workstation

```shell
export MUJOCO_GL=osmesa
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
export MJLIB_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
```

if on a MAC OSX

```shell
export MUJOCO_GL=glfw
export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.dylib
export MJLIB_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
```

## Running the script that performs the pretraining and training process
To run the main loops one needs to activate the environment and then run:
```shell
python train.py <env_DMC>
```
<env_DMC> argument should be one of the following:
* acrobot_swingup
* finger_turn_hard
* walker_run
* cheetah_run
* hopper_hop
* humanoid_run
* quadruped_walk 
* quadruped_run

## Plotting
In order to plot the rewards for a certain environments, one first needs to run the main loops for the pretraining/training.
After that:
```shell
python plot.py <env_DMC>
```
<env_DMC> argument should be one of the following:
* acrobot_swingup
* finger_turn_hard
* walker_run
* cheetah_run
* hopper_hop
* humanoid_run
* quadruped_run
* quadruped_walk

The output of the "plot.py" script is a .pdf file (with the name of the DMC environment) that contains the plot for the rewards for all episodes of the training process

## Features
Files Descriptions:
* <u>conda_env.yml</u> : Contains the packages that are required for the repo.
* <u>config.yml</u> : Contains all the arguments for the repo
* <u>replay_buffer.py</u> : The file that contains the class implementation of the Experience Replay.
* <u>utils.py</u> : Contains useful functions and classes, such as the class of the MLP network, function for wrapping the enviroment, function for soft updates of actor networks, function for setting the seeds and function initializing the weights.
* <u>Agent.py</u> : Contains the implementation of the Actor class, the implementation of the Critic class and the implementation of the Agent class.
* <u>train.py</u> : The file that performs the main training process.
