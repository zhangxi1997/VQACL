# VQACL: A Novel Visual Question Answering Continual Learning Setting

We establish a novel VQA Continual Learning setting named VQACL, which contains two key components: a dual-level task sequence where visual and linguistic data are nested, and a novel composition testing containing new skill-concept combinations. The former devotes to simulating the ever-changing multimodal datastream in real world and the latter aims at measuring models’ generalizability for cognitive reasoning.

To do the VQACL, we also propose a novel representation learning method, which leverages a sample-specific and a sample-invariant feature to learn
representations that are both discriminative and generalizable for VQA.

## Setup
```bash
# Create python environment (optional)
conda create -n vqacl python=3.7
source activate vqacl

# Install python dependencies
pip install -r requirements.txt

# Download T5 backbone checkpoint
python download_backbones.py

```

## Code structure
```bash
# Store images, features, and annotations
./datasets
    COCO/
        images/
        featuers/
    vqa/
        Paritition_Q/
    ...


# Training and testing in the VQACL setting
./VL-T5/
    src/
        modeling_t5_our.py                                    <= Our VL-T5 model classes
        vqacl.py vqacl_comp.py vqa_data.py vqa_model.py ...   <= Testing in the VQACL setting
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for evaluation
```

## Dataset Preparation / Model checkpoint
- Download `datasets/COCO` and `datasets/vqa` from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)
- Download model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1GDI9uG9OSQk0ObEaEJI3C6eKcDzh0yGp?usp=share_link)

## VQACL tasks

```bash
# Training with 1 gpu
cd VL-T5/
bash scripts/VQACL_train.sh 1 # Standard Training
bash scripts/VQACL_COMP_train.sh 1 # Training for Novel Composition Testing (Group-5)

# Testing with 1 gpu
cd VL-T5/
bash scripts/VQACL.sh 1 # Standard Testing
bash scripts/VQACL_COMP.sh 1 # Novel Composition Testing (Group-5)
```

## Acknowledgement

Our model is based on the official [VL-T5](https://github.com/j-min/VL-T5) repository, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.
