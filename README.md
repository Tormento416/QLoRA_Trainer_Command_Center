# Rez_SLM

Resume Writing, Candidate Screening & Fine-Tuning Small Language Model (SLM) Architecture for Local Inference and QLoRA Training.

## Project Features
- **Local SLM Loader**: CUDA acceleration on NVIDIA RTX GPUs with 4-bit `bitsandbytes` quantization.
- **Conversation Memory Bank**: Short-term sliding context window & persistent long-term fact storage (`data/memory_bank.json`).
- **Universal Dataset Importer**: Smart auto-scanning importer for Kaggle, Hugging Face, local ZIP archives, CSV, JSON, and Parquet datasets.
- **QLoRA Fine-Tuning Pipeline**: 4-bit LoRA adapter training (`bf16` precision) with support for Gemma 4 architectures.
