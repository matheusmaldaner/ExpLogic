# eXpLogic

<div align="center">
  <a href="https://scholar.google.com/citations?user=erXnlb4AAAAJ&hl=en">
    <img src="https://i.imgur.com/3X52Var.png" alt="eXpLogic Framework" width="500" style="background-color: white; padding: 20px; border-radius: 10px;">
  </a>
</div>

eXpLogic is an algorithm designed to generate saliency maps for explaining input patterns and logic behaviors in DiffLogic networks. It provides insights into how certain inputs influence predictions, supports model debugging, and enables resource-efficient inference through MiniNets. The project leverages the DiffLogic architecture to promote transparency and interpretability in AI systems.

## Deployment Procedure

```bash
conda create --prefix=./explogic_env python=3.9 -y
conda activate explogic_env
conda install pip
pip install ipykernel
python -m ipykernel install --user --name=explogic_env --display-name="EXPLOGIC"
module load cuda/11.1.0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
mamba install cudatoolkit=11.1
pip install difflogic
pip install -r requirements.txt
```

## Features

1. **Local Explanations**: Highlights specific input contributions to individual predictions.
2. **Function Explanations**: Visualizes patterns that activate functions in the DiffLogic network.
3. **MiniNet Optimization**: Reduces network size and inference time for class-specific predictions.
4. **SwitchDist Metric**: Evaluates saliency maps by measuring input changes required to alter predictions.

## Usage

- **Input Data**: Provide datasets in the required format for classification tasks (e.g., MNIST, FMNIST).
- **Training Models**: Follow the training scripts to create and evaluate DiffLogic networks.
- **Explanation Generation**: Use the provided tools to generate saliency maps and MiniNet models.

## Common Errors

- Ensure you are using Numpy 1.x, as some modules are not compatible with Numpy 2.x.

## Citation

If you use eXpLogic in your research, please cite the corresponding paper:

_Wormald, S., Koblah, D., Maldaner, M. K., Forte, D., & Woodard, D. L. (2024). eXpLogic: Explaining Logic Types and Patterns in DiffLogic Networks. Springer Journal._

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.
