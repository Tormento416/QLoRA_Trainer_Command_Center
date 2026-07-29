import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    Gemma4ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import time
from transformers import TrainerCallback


class CustomControlCallback(TrainerCallback):
    def __init__(self, stop_checker=None, progress_callback=None, max_minutes=0, total_dataset_records=4940000):
        self.stop_checker = stop_checker
        self.progress_callback = progress_callback
        self.max_minutes = max_minutes
        self.total_dataset_records = total_dataset_records
        self.start_time = time.time()
        self.last_heartbeat_time = time.time()
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_checker and self.stop_checker():
            print("\n[QLoRA] Stop requested by user! Halting training gracefully...")
            control.should_training_stop = True
        if self.max_minutes > 0:
            elapsed_mins = (time.time() - self.start_time) / 60.0
            if elapsed_mins >= self.max_minutes:
                print(f"\n[QLoRA] Max training time limit reached ({self.max_minutes} mins)! Halting training...")
                control.should_training_stop = True

        # Retrieve latest loss value from state log history
        loss_val = 0.0
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    try:
                        loss_val = float(entry["loss"])
                    except (ValueError, TypeError):
                        loss_val = 0.0
                    break

        eff_batch = (args.per_device_train_batch_size or 1) * (args.gradient_accumulation_steps or 1)
        records_processed = state.global_step * eff_batch
        pct_4_94m = (records_processed / float(self.total_dataset_records)) * 100.0

        if self.progress_callback:
            try:
                self.progress_callback(state.global_step, state.max_steps, loss_val, self.start_time, eff_batch)
            except TypeError:
                self.progress_callback(state.global_step, state.max_steps, loss_val, self.start_time)

        # 60-Second Periodic Status Heartbeat Log
        now = time.time()
        if now - self.last_heartbeat_time >= 60.0:
            self.last_heartbeat_time = now
            steps_done = state.global_step - self.last_step
            self.last_step = state.global_step
            sec_per_it = 60.0 / max(1, steps_done)
            pct_run = (state.global_step / max(1, state.max_steps)) * 100.0 if state.max_steps > 0 else 0.0

            elapsed_sec = int(now - self.start_time)
            m, s = divmod(elapsed_sec, 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

            vram_str = ""
            if torch.cuda.is_available():
                allocated = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
                reserved = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
                vram_str = f" | Actual GPU VRAM consumed={allocated} GB | Total VRAM reserved={reserved} GB"

            print(f"[Heartbeat {time.strftime('%H:%M:%S')}] Step {state.global_step}/{state.max_steps:,} ({pct_run:.1f}% Run | {pct_4_94m:.3f}% of 4.94M Dataset) | Processed: {records_processed:,}/4,940,000 records | Loss: {loss_val:.4f} | Pace: {sec_per_it:.1f}s/it | Elapsed: {time_str}{vram_str} | Status: RUNNING OK")



def run_qlora_training(
    config_path: str = "config.yaml",
    dataset_path: str = "Finetune/training_data.jsonl",
    output_dir: str = "Finetune/output_adapter",
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = 4500,
    sub_sample: int = 45000,
    max_length: int = 384,
    r: int = 16,
    lora_alpha: int = 32,
    dataloader_num_workers: int = 0,
    max_minutes: int = 0,
    mixed_precision: str = "Auto",
    gradient_checkpointing: bool = True,
    grad_accum: int = 0,
    stop_checker_fn=None,
    progress_callback_fn=None,
    model_path: str = None,
):
    print("\n" + "=" * 50)
    print("  High-Performance QLoRA Fine-Tuning Pipeline")
    print("=" * 50)

    # Throttling CPU thread priority to ensure smooth OS performance
    if sys.platform == "win32":
        try:
            import psutil
            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("[QLoRA] CPU Process Priority set to BELOW_NORMAL to maintain smooth OS responsiveness.")
        except Exception:
            pass

    # 1. Load configuration for model path
    base_model_path = model_path if model_path else "d:/models"
    if not model_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            base_model_path = cfg.get("model", {}).get("base_path", base_model_path)

    print(f"[QLoRA] Base Model Path = {base_model_path}")
    print(f"[QLoRA] Training Dataset = {dataset_path}")
    print(f"[QLoRA] Output Directory = {output_dir}")
    print(f"[QLoRA] Config Profile: Batch Size = {batch_size} | Sub-sample = {sub_sample:,} | Max Steps = {max_steps} | Max Length = {max_length}")

    # 2. Check GPU & Quantization & Enable TF32 Tensor Cores
    is_cuda = torch.cuda.is_available()
    print(f"[QLoRA] CUDA Available: {is_cuda}")
    if is_cuda:
        print(f"[QLoRA] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    bnb_config = None
    compute_dtype = torch.float16
    
    if mixed_precision == "BF16" or (mixed_precision == "Auto" and is_cuda and torch.cuda.is_bf16_supported()):
        compute_dtype = torch.bfloat16
    elif mixed_precision == "FP16":
        compute_dtype = torch.float16

    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # 3. Load Tokenizer & Model
    print("[QLoRA] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"

    print("[QLoRA] Loading base model for 4-bit k-bit training...")
    model_kwargs = {"device_map": {"": 0}} if is_cuda else {"device_map": "cpu", "dtype": torch.float32}
    if is_cuda:
        model_kwargs["attn_implementation"] = "sdpa"
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = Gemma4ForConditionalGeneration.from_pretrained(base_model_path, **model_kwargs)  # was AutoModelForCausalLM

    if is_cuda:
        model = prepare_model_for_kbit_training(model)

    # 4. Set up LoRA Configuration
    # Gemma 4 E4B-it is multimodal — scope LoRA to text-decoder layers only,
    # bare "linear"/"q_proj" leaks into vision/audio towers otherwise.
    target_mods = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
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

    # 5. Load Dataset & Optional Sub-sampling
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    if sub_sample and sub_sample > 0 and len(dataset) > sub_sample:
        print(f"[QLoRA] Sub-sampling dataset from {len(dataset):,} records down to {sub_sample:,} records...")
        dataset = dataset.shuffle(seed=42).select(range(sub_sample))

    print(f"[QLoRA] Final Dataset size for training: {len(dataset):,} records")

    def format_and_tokenize(example):
        messages = example.get("messages", [])
        if not messages:
            return {"input_ids": [], "labels": []}

        if getattr(tokenizer, "chat_template", None) is not None:
            try:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                formatted_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        else:
            formatted_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

        tokenized = tokenizer(formatted_text, truncation=True, max_length=max_length, padding=False)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Limit CPU tokenization workers to half CPU count so Windows UI never stutters
    cpu_count = os.cpu_count() or 2
    num_proc = min(4, max(1, cpu_count // 2)) if len(dataset) > 1000 else 1
    print(f"[QLoRA] Tokenizing dataset using {num_proc} parallel CPU processes (smooth OS priority)...")
    tokenized_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing dataset"
    )

    # 6. Training Arguments
    calculated_grad_accum = grad_accum if grad_accum > 0 else (3 if batch_size == 6 else (4 if batch_size <= 4 else 2))
    
    use_bf16 = False
    use_fp16 = False
    if is_cuda:
        if mixed_precision == "BF16" or (mixed_precision == "Auto" and torch.cuda.is_bf16_supported()):
            use_bf16 = True
        elif mixed_precision == "FP16" or (mixed_precision == "Auto" and not torch.cuda.is_bf16_supported()):
            use_fp16 = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=calculated_grad_accum,
        learning_rate=learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        num_train_epochs=epochs if max_steps <= 0 else 100,
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=1,
        save_strategy="steps" if max_steps > 0 else "epoch",
        save_steps=500 if max_steps > 0 else 500,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=is_cuda,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    control_callback = CustomControlCallback(
        stop_checker=stop_checker_fn,
        progress_callback=progress_callback_fn,
        max_minutes=max_minutes
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[control_callback],
    )

    print("[QLoRA] Starting high-performance training run...")
    trainer.train()

    print(f"[QLoRA] Saving fine-tuned LoRA adapter to {output_dir}...")
    try:
        import huggingface_hub
        tmpl_dir = os.path.join(os.path.dirname(huggingface_hub.__file__), "templates")
        os.makedirs(tmpl_dir, exist_ok=True)
        for t_file in ["modelcard_template.md", "datasetcard_template.md"]:
            t_path = os.path.join(tmpl_dir, t_file)
            if not os.path.exists(t_path):
                with open(t_path, "w", encoding="utf-8") as f:
                    f.write("---\n# Model Card\n---\n")
    except Exception as e_tmpl:
        print(f"[QLoRA] ModelCard template check note: {e_tmpl}")

    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[QLoRA] Fine-tuning complete!")
    except Exception as e_save:
        print(f"[QLoRA] Warning during adapter save: {e_save}")
        try:
            model.save_pretrained(output_dir, create_model_card=False)
            tokenizer.save_pretrained(output_dir)
            print("[QLoRA] Fine-tuning adapter saved successfully (fallback mode).")
        except Exception as e_fallback:
            print(f"[QLoRA] Secondary save error: {e_fallback}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuner for Local SLM")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", default="Finetune/training_data.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output_dir", default="Finetune/output_adapter", help="Directory to save LoRA weights")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU (default 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=4500, help="Max training steps (e.g. 4500)")
    parser.add_argument("--sub_sample", type=int, default=45000, help="Sub-sample dataset to N records (e.g. 45000)")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length truncation (e.g. 384)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader worker processes (default 4)")
    args = parser.parse_args()

    run_qlora_training(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        sub_sample=args.sub_sample,
        max_length=args.max_length,
        dataloader_num_workers=args.dataloader_num_workers,
    )