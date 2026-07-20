import os
import argparse
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def run_qlora_training(
    config_path: str = "config.yaml",
    dataset_path: str = "Finetune/dataset_template.jsonl",
    output_dir: str = "Finetune/output_adapter",
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    r: int = 16,
    lora_alpha: int = 32,
):
    print("\n" + "=" * 50)
    print("  QLoRA Fine-Tuning Pipeline for Local SLM")
    print("=" * 50)

    # 1. Load configuration for model path
    base_model_path = "d:/models"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            base_model_path = cfg.get("model", {}).get("base_path", base_model_path)

    print(f"[QLoRA] Base Model Path = {base_model_path}")
    print(f"[QLoRA] Training Dataset = {dataset_path}")
    print(f"[QLoRA] Output Directory = {output_dir}")

    # 2. Check GPU & Quantization
    is_cuda = torch.cuda.is_available()
    print(f"[QLoRA] CUDA Available: {is_cuda}")
    if is_cuda:
        print(f"[QLoRA] GPU: {torch.cuda.get_device_name(0)}")

    bnb_config = None
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # 3. Load Tokenizer & Model
    print("[QLoRA] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[QLoRA] Loading base model for 4-bit k-bit training...")
    model_kwargs = {"device_map": "auto"} if is_cuda else {"device_map": "cpu", "dtype": torch.float32}
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    if is_cuda:
        model = prepare_model_for_kbit_training(model)

    # 4. Set up LoRA Configuration (targeting Gemma 4 4-bit linear submodules)
    target_mods = ["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "gate_proj.linear", "up_proj.linear", "down_proj.linear"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_mods,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_and_tokenize(example):
        messages = example.get("messages", [])
        if not messages:
            return {"input_ids": [], "labels": []}

        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(formatted_text, truncation=True, max_length=512, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=is_cuda,
        fp16=False,
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("[QLoRA] Starting training...")
    trainer.train()

    print(f"[QLoRA] Saving fine-tuned LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[QLoRA] Fine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuner for Local SLM")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", default="Finetune/dataset_template.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output_dir", default="Finetune/output_adapter", help="Directory to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    run_qlora_training(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
