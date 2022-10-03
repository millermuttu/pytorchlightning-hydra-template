<div align="center">

# Session 4: Deployments for demo

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/ashleve/lightning-hydra-template/graphs/contributors)

<!-- <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a> -->

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.2-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a> -->



</div>

## Description

Deployments for demo: in this session we will go from research to prodcution of our pytorch project, where we will be training CIFAR10 model using pytorch lighting and hydra, once the model is ready a inference engine is made using Gradio and docker image is build to run the inference engine.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/millermuttu/pytorchlightning-hydra-template.git
cd pytorchlightning-hydra-template
git checkout session4

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with example configuration

```bash
# train on CPU
python src/train.py experiment=cifar trainer=cpu

# train on GPU
python src/train.py experiment=cifar trainer=gpu
```

once the training is done Scripted model will be saved under the traini logging. this model can be served without dependency of the python, even to other languages like C/C++, java etc.

Tracing the model:

```
log.info("tracing model...")
example = torch.rand(1,3, 32,32)
model.eval()
scrited_model = torch.jit.trace(model, example)
torch.jit.save(scrited_model, f"{cfg.paths.output_dir}/model.trace.pt")
log.info(f"Saving traced model to {cfg.paths.output_dir}/model.trace.pt")

```

this saved model is used for inferencing the model using demo app, here we need to give demo_scripted.yaml as the config file to the hydra

```
python src/demo_scripted.py ckpt_path="path to saved model"
```

## Converting the demo app to docker images

### Create the docker image
```
FROM python:3.8.13-slim-bullseye

RUN export DEBIAN_FRONTEND=noninteractive \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
  && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
  && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
  && apt update && apt install -y locales \
  && locale-gen en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install \
  torch==1.12.0+cpu \
  torchvision==0.13.0+cpu \
  gradio==3.3.0 \
  timm \
  pyrootutils \
  hydra-core==1.2.0 \
  hydra-colorlog==1.2.0 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  && rm -rf /root/.cache/pip

# WORKDIR /workspace/project
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY .git pyproject.toml ./
EXPOSE 8080

CMD ["python", "src/demo_scripted.py", "-p 8080:8080"]
```

### Build the docker image 
```bash
dokcer build -f Dockerfile.demo.scripted.onestage -t session4:latest .
```

here since we are using multiple dockerfile to create diffent images we need to specify the dockerfile using **-f** notation

### Run the docker image
```bash
docker run -t session4:latest
```

Thsi will host the app in localhost and you canrun and test the model using the localhost site.
