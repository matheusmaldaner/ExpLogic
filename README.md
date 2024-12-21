# ExpLogic

## Deployment on HiPerGator

```
module load conda
conda create --prefix=./difflogic_env python=3.9 -y
conda activate difflogic_env
conda install pip
pip install ipykernel
python -m ipykernel install --user --name=difflogic_env --display-name="DIFFLOGIC"
module load cuda/11.1.0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html
mamba install cudatoolkit=11.1 
pip install difflogic
pip install -r requirements.txt
```

## Common Errors
- Ensure you are using Numpy 1.x as some modules do not yet support Numpy 2.x
