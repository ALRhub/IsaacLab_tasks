#!/bin/zsh

# Initialize Conda for this script
eval "$(conda shell.zsh hook)"

# Define environment name and Python version
ENV_NAME=isaaclab
PYTHON_VERSION=3.10

# Create a new conda environment
echo "Creating a new conda environment named $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate the newly created environment
echo "Activating the $ENV_NAME environment"
conda activate $ENV_NAME

# Verify if the correct conda environment is activated
echo
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    echo Failed to activate conda environment.
    exit 1
else
    echo Successfully activated conda environment.
fi

# Install mamba release to boost installation and resolve dependencies
conda install -c conda-forge mamba=1.4.2 -y


# Install packages using conda or mamba
echo "Installing packages with conda or mamba"
conda install -c hussamalafandi cppprojection -c conda-forge -y

mamba install pytorch=2.2.1 torchvision=0.17.1 torchaudio=2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install conda-forge::wandb=0.16.3 -y
mamba install conda-forge::natsort=8.4.0 -y
mamba install conda-forge::tabulate=0.9.0 -y
mamba install conda-forge::conda-build=24.1.2 -y
mamba install conda-forge::mp_pytorch=0.1.4 -y
mamba install conda-forge::cw2=2.5.1 -y


# ----- Install IsaacSim -----
echo "Installing IsaacSim"
pip install --upgrade pip
# pip install warp-lang
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install isaacsim isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

# ----- Install IsaacLab -----
echo "Installing IsaacLab"
git clone -b alr_tasks git@github.com:FAR3L/IsaacLab.git
cd IsaacLab
conda develop .
./isaaclab.sh -i
cd ..

# ----- Install other packages from forks -----

# Fancy gym
git clone -b isaaclab git@github.com:FAR3L/fancy_gym.git
cd fancy_gym
conda develop .
pip install -e .
cd ..

# stable-baselines3
git clone -b alr_tasks git@github.com:FAR3L/stable-baselines3.git
cd stable-baselines3
conda develop .
pip install -e .
cd ..

# alr_tasks
git clone -b box_pushing git@github.com:ALRhub/IsaacLab_tasks.git alr_tasks
cd alr_tasks/exts/alr_isaaclab_tasks
conda develop .
pip install -e .
cd ....

# pytorch kinematics
pip install pytorch-kinematics

# ----- Install TCE -----

# TCE
git clone -b isaaclab git@github.com:FAR3L/TCE_RL.git
cd TCE_RL
conda develop .
cd ..

# Trust_Region_Projection
git clone -b TCE_ICLR24 --single-branch git@github.com:BruceGeLi/trust-region-layers.git
cd trust-region-layers
conda develop .
cd ..

# Git_Repo_Tracker
git clone -b main --single-branch git@github.com:ALRhub/Git_Repos_Tracker.git
cd Git_Repos_Tracker
pip install -e .
conda develop .
cd ..

# MetaWorld
git clone -b tce_final --single-branch git@github.com:BruceGeLi/Metaworld.git
cd Metaworld
pip install -e .
conda develop .
cd ..

echo "Configuration completed successfully."