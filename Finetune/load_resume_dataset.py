import os
import json
import pandas as pd
import kagglehub

print("[Kaggle] Downloading dataset 'snehaanbhawal/resume-dataset'...")
folder_path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
csv_file_path = os.path.join(folder_path, "Resume", "Resume.csv")

if os.path.exists(csv_file_path):
    print(f"[Kaggle] Reading CSV from '{csv_file_path}'...")
    df = pd.read_csv(csv_file_path)
    print(f"[Kaggle] Loaded {len(df)} resume records.")

    output_jsonl_path = "Finetune/resume_dataset.jsonl"
    print(f"[Kaggle] Converting into fine-tuning dataset: '{output_jsonl_path}'...")

    records = []
    for idx, row in df.iterrows():
        category = str(row.get("Category", "Professional")).strip()
        resume_text = str(row.get("Resume_str", "")).strip()

        if not resume_text:
            continue

        # Format into chat template for fine-tuning
        user_prompt = f"Analyze the following resume and identify its job category:\n\n{resume_text[:1000]}"
        assistant_response = f"Job Category: {category}\nKey Qualifications Summary: {resume_text[:300]}..."

        entry = {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        }
        records.append(entry)

    os.makedirs("Finetune", exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Kaggle] Successfully generated '{output_jsonl_path}' with {len(records)} records!")
    print(f"\n[QLoRA Command]: python Finetune/qlora_trainer.py --dataset {output_jsonl_path} --epochs 3")
else:
    print("[Kaggle] ERROR: Resume.csv not found in downloaded dataset.")
