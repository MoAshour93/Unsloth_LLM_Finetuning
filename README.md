# 🔧 Finetuning Llama 3.1 8B Instruct Model for RICS APC, Bid Responses, and More 📚

This repository contains the code and steps required to fine-tune the **Llama 3.1 8B Instruct model** using [Unsloth library 🚀](https://github.com/unsloth). This fine-tuned model is designed to aid RICS APC candidates with drafting submissions, support bid managers in crafting compelling bid responses, enhance CV writing, and streamline Bills of Quantities codification. The result is a powerful, domain-specific LLM capable of transforming various documentation needs in the construction industry.

## 📑 Table of Contents

- [📋 Project Overview](#-project-overview)
  - [🎯 Objective](#-objective)
  - [🗝️ Key Steps](#-key-steps)
- [⚙️ Setup and Installation](#-setup-and-installation)
  - [1️⃣ Environment Setup](#-1-environment-setup)
    - [🔧 Step 1.1: Windows Subsystem for Linux](#-step-11-windows-subsystem-for-linux)
    - [🐍 Step 1.2: Anaconda](#-step-12-anaconda)
    - [📦 Step 1.3: Installing Unsloth](#-step-13-installing-unsloth)
    - [💻 Step 1.4: Installing PyTorch and Nvidia CUDA Toolkit](#-step-14-installing-pytorch-and-nvidia-cuda-toolkit)
  - [📂 2. Dataset Preparation](#-2-dataset-preparation)
  - [🧩 3. Model Loading and Configuration](#-3-model-loading-and-configuration)
  - [🔍 4. Model Training and Quantization](#-4-model-training-and-quantization)
  - [🌐 5. Local Deployment with OpenWebUI](#-5-local-deployment-with-openwebui)
- [🚀 Usage](#-usage)
- [🔮 Future Enhancements](#-future-enhancements)
- [💬 Support and Contributions](#-support-and-contributions)
- [🔗 General Links & Resources](#-general-links--resources)

---

## 📋 Project Overview

### 🎯 Objective
This project fine-tunes the **Llama 3.1 8B Instruct model** to create a tailored language model designed to assist RICS APC candidates, bid managers, CV writers, and Quantity Surveyors. The fine-tuned model provides an intelligent assistant that can:
- Help **RICS APC candidates** craft structured, impactful responses using previous successful submissions.
- Assist **bid managers** in formulating compelling bid responses, increasing chances of winning projects.
- Enhance **CV writing** by generating client-ready CVs that showcase relevant skills and experiences effectively.
- Simplify **Bills of Quantities (BoQs) codification** by aligning with specific measurement standards, reducing manual effort.

### 🗝️ Key Steps
1. **Environment Setup** 🖥️: Install dependencies and configure necessary libraries.
2. **Dataset Transformation** 📊: Convert and format the dataset into **ShareGPT** and **Huggingface** formats.
3. **Model Loading and Configuration** ⚙️: Load Llama 3.1 8B Instruct model and set up fine-tuning parameters.
4. **Model Training and Quantization** 🎓: Fine-tune the model and quantize for efficient storage.
5. **Deployment** 🌐: Deploy the fine-tuned model locally using OpenWebUI.

---

## ⚙️ Setup and Installation

### 1️⃣ Environment Setup

#### 🔧 Step 1.1: Windows Subsystem for Linux
- Use a Linux-based system for Unsloth. Download your preferred distribution [here](https://www.linux.org/pages/download/), or follow [Microsoft’s WSL guide](https://learn.microsoft.com/en-us/windows/wsl/install).

      wsl --install

#### 🐍 Step 1.2: Anaconda
- Download **Anaconda** from [Anaconda’s main site](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh), then execute:

      chmod +x Anaconda3-[version]-Linux-x86_64.sh
      ./Anaconda3-[version]-Linux-x86_64.sh

  Replace `[version]` as needed.

#### 📦 Step 1.3: Installing Unsloth
- Follow the setup steps on the [Unsloth repository](https://github.com/unslothai/unsloth?tab=readme-ov-file), or open a terminal and run:

      conda create unsloth_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
      conda activate unsloth_env
      pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
      pip install --no-deps "trl<0.9.0" wandb huggingface_hub peft accelerate bitsandbytes datasets

#### 💻 Step 1.4: Installing PyTorch and Nvidia CUDA Toolkit
- Install **PyTorch** with CUDA:

      pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

- Install **Nvidia CUDA Toolkit v12.1**:

      wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
      sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
      wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
      sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.0-1_amd64.deb
      sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
      sudo apt-get update
      sudo apt-get -y install cuda

### 📂 2. Dataset Preparation

Convert the anonymized dataset into [ShareGPT format](https://huggingface.co/docs/datasets/sharegpt) for APC submissions, bid responses, CVs, and BoQs codification data, then transform it to Huggingface format.

- Load the dataset.
- Convert to ShareGPT JSON format.
- Prepare for Huggingface compatibility.

### 🧩 3. Model Loading and Configuration

Load **Llama 3.1 8B Instruct model** with Unsloth and configure parameters like **LoRA** for efficient training, learning rates, and batch sizes.

Refer to the notebook [`Finetuning_RICS_APC_LLM_llama3_1_instruct_8b_conversational.ipynb`](./notebooks/Finetuning_RICS_APC_LLM_llama3_1_instruct_8b_conversational.ipynb) for details.

### 🔍 4. Model Training and Quantization

1. Run the training scripts to fine-tune the model.
2. Quantize post-training for efficient storage and performance.

The model is saved under `models/`.

### 🌐 5. Local Deployment with OpenWebUI

1. Install **OpenWebUI** dependencies.
2. Load the quantized model to OpenWebUI for an interactive chatbot experience.

---

## 🚀 Usage

- **RICS APC Candidates**: Generate tailored submission drafts with structured responses.
- **Bid Managers**: Formulate winning bid responses for securing projects.
- **CV Writers**: Enhance client CVs, highlighting relevant skills and experiences.
- **Quantity Surveyors**: Expedite BoQ codification in compliance with measurement standards.

---

## 🔮 Future Enhancements

Possible upgrades:
- Utilize larger datasets and explore data augmentation.
- Offer cloud deployment for remote usage.
- Add feedback mechanisms for enhanced usability.

---

## 💬 Support and Contributions

Contributions are welcome! For substantial changes, please open an issue. Connect via our [LinkedIn Page](https://www.linkedin.com/company/apc-mastery-path) or [website](https://www.apcmasterypath.co.uk).

---

## 🔗 General Links & Resources

- **Our Website**: [www.apcmasterypath.co.uk](https://www.apcmasterypath.co.uk)
- **APC Mastery Path Blogposts**: [APC Blogposts](https://www.apcmasterypath.co.uk/blog-list)
- **LinkedIn Pages**: [Personal](https://www.linkedin.com/in/mohamed-ashour-0727/) | [APC Mastery Path](https://www.linkedin.com/company/apc-mastery-path)
