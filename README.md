# VQACL: A Novel Visual Question Answering Continual Learning Setting


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
# Testing with 4 gpus
cd VL-T5/
bash scripts/VQACL.sh 1 # Standard Testing
bash scripts/VQACL_COMP.sh 1 # Novel Composition Testing (Group-5)
```
