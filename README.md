# ExpLogic

## Deployment Proceedure

```
module load conda
conda create --prefix=./explogic_env python=3.9 -y
conda activate explogic_env
conda install pip
pip install ipykernel
python -m ipykernel install --user --name=explogic_env --display-name="EXPLOGIC"
module load cuda/11.1.0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111  -f https://download.pytorch.org/whl/torch_stable.html
mamba install cudatoolkit=11.1 
pip install difflogic
pip install -r requirements.txt
```

## Common Errors
- Ensure you are using Numpy 1.x as some modules do not yet support Numpy 2.x
