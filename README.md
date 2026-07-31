# Generic QLoRA Trainer
A powerful, standalone desktop application for fine-tuning Local Language Models (SLMs/LLMs) using high-performance QLoRA on local consumer hardware.

(Built by A. Sousa/Tormento416)

## Features
- **Generic Architecture Support:** Fine-tune almost any causal language model (LLaMA 3, Mistral, Qwen, Phi, Gemma).
- **Cross-GPU Intelligence:** Dynamically detects NVIDIA CUDA, AMD ROCm, and Apple Silicon MPS.
- **Stable Training:** Uses advanced hyperparameter injection (NEFTune noise) to prevent sudden loss spikes.
- **Smart Pathing:** Automatically remembers your preferred model and dataset paths across sessions.

## 🚀 Getting Started

### For Windows Users
The easiest way to use this application is via the pre-compiled standalone package.
1. Open the `Windows_Standalone_App` folder.
2. Double click `QLoRA_Trainer.exe`.
*No Python installation or dependencies required!*

### For Mac Users (Apple Silicon / MPS)
Since the app needs to be compiled natively for MacOS, you can use our built-in Mac Compiler Kit:
1. Open your Mac Terminal and navigate to the `source_code` folder in this repo.
2. Run the build script:
   ```bash
   sh build_mac.sh
   ```
3. This script will install dependencies and use PyInstaller to build a native Mac Application. You will find your compiled app in the `dist/` folder!

### For Developers (Running from Source)
If you want to run the python scripts directly or modify the code:
1. Install Python 3.10+
2. Navigate to the `source_code` directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the trainer:
   ```bash
   python QLoRA_Trainer.py
   ```
### For Linux & Cloud Server Users
Because the core architecture is built in pure Python, it runs natively (and exceptionally well) on Linux machines! In fact, running this on a cloud GPU server (like AWS, RunPod, or Lambda Labs) running Linux is highly recommended because `BitsAndBytes` is optimized specifically for Linux.
1. Connect to your Cloud GPU terminal.
2. Clone this repository:
   ```bash
   git clone https://github.com/Tormento416/QLoRA_Trainer_Command_Center.git
   ```
3. Install dependencies:
   ```bash
   cd QLoRA_Trainer_Command_Center/source_code
   pip install -r requirements.txt
   ```
4. If your server is purely headless (no desktop GUI), you can bypass the `QLoRA_Trainer.py` interface and run the `Finetune/generic_qlora_trainer.py` backend script directly from the terminal!

*(Note: You cannot train a model over a network URL. The model's raw weights must be physically present on the same machine/server where the GPU is doing the math!)*

## Dataset Format
The trainer strictly requires your dataset to be in standard conversational JSONL format. Ensure your `training_data.jsonl` lines look exactly like this:
```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there."}]}
```
