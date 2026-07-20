import os
import glob
import json
import zipfile
import tempfile
import argparse
import pandas as pd

COMMON_TARGET_KEYWORDS = [
    "hired", "target", "label", "class", "category", "answer", "response",
    "output", "decision", "status", "hires", "score", "rating", "result",
    "verdict", "job_title", "sentiment", "summary", "jobrole"
]

COMMON_INPUT_KEYWORDS = [
    "prompt", "question", "instruction", "text", "resume", "input",
    "description", "content", "document", "body", "features", "profile", "str_resume"
]

def append_to_jsonl(records, output_file="Finetune/training_data.jsonl"):
    if not records:
        return 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)

def auto_detect_columns(df):
    cols_lower = {col.lower(): col for col in df.columns}
    target_col = None
    input_col = None

    for kw in COMMON_TARGET_KEYWORDS:
        for cl, orig in cols_lower.items():
            if cl == kw or kw in cl:
                target_col = orig
                break
        if target_col:
            break

    for kw in COMMON_INPUT_KEYWORDS:
        for cl, orig in cols_lower.items():
            if cl == kw or (kw in cl and orig != target_col):
                input_col = orig
                break
        if input_col:
            break

    if not target_col and len(df.columns) > 1:
        target_col = df.columns[-1]

    if not input_col:
        input_col = "*"

    return input_col, target_col

def process_dataframe(df, user_col="auto", assistant_col="auto", system_prompt=None):
    if df.empty:
        return []

    if not user_col or user_col == "auto" or not assistant_col or assistant_col == "auto":
        det_user, det_target = auto_detect_columns(df)
        if not user_col or user_col == "auto":
            user_col = det_user
        if not assistant_col or assistant_col == "auto":
            assistant_col = det_target

    print(f"[AutoScan] Auto-mapped Columns -> Prompt: '{user_col}' | Target Response: '{assistant_col}'")

    records = []
    is_wildcard = user_col in ["*", "auto", "all"] or "," in user_col

    for idx, row in df.iterrows():
        assistant_val = str(row.get(assistant_col, "")).strip() if assistant_col in row else ""
        if not assistant_val or assistant_val.lower() == "nan":
            continue

        if is_wildcard:
            feature_cols = [c for c in df.columns if c != assistant_col and c.lower() != "candidate_id" and c.lower() != "id"]
            profile_lines = [f"- {col}: {row[col]}" for col in feature_cols if pd.notna(row[col])]
            user_val = "Analyze the following record attributes:\n" + "\n".join(profile_lines)
        else:
            user_val = str(row.get(user_col, "")).strip() if user_col in row else ""

        if not user_val or user_val.lower() == "nan":
            continue

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_val})
        messages.append({"role": "assistant", "content": f"{assistant_col}: {assistant_val}" if assistant_col else assistant_val})

        records.append({"messages": messages})
    return records

def process_single_file(file_path, user_col="auto", assistant_col="auto", system_prompt=None):
    if file_path.lower().endswith(".zip"):
        print(f"[AutoScan] ZIP Archive detected: '{file_path}'. Extracting and scanning contents...")
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            records = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl", ".txt")):
                        extracted_file = os.path.join(root, f)
                        records.extend(process_single_file(extracted_file, user_col, assistant_col, system_prompt))
            return records
        except Exception as e:
            print(f"[AutoScan] Failed to extract zip '{file_path}': {e}")
            return []

    print(f"[AutoScan] Scanning file: {file_path}")
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"text_content": lines})
            user_col = "auto"
            assistant_col = "text_content"
        else:
            return []

        print(f"[AutoScan] Loaded shape: {df.shape}")
        return process_dataframe(df, user_col, assistant_col, system_prompt)
    except Exception as e:
        print(f"[AutoScan] Warning: Failed to process '{file_path}': {e}")
        return []

def scan_path_pattern(path_pattern, user_col="auto", assistant_col="auto", output_file="Finetune/training_data.jsonl", system_prompt=None):
    matched_paths = glob.glob(path_pattern, recursive=True)
    if not matched_paths and os.path.exists(path_pattern):
        matched_paths = [path_pattern]

    print(f"[AutoScan] Found {len(matched_paths)} matching items for pattern '{path_pattern}'...")
    total_added = 0

    for path in matched_paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith((".csv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl", ".txt", ".zip")):
                        full_p = os.path.join(root, f)
                        records = process_single_file(full_p, user_col, assistant_col, system_prompt)
                        added = append_to_jsonl(records, output_file)
                        total_added += added
        elif os.path.isfile(path):
            records = process_single_file(path, user_col, assistant_col, system_prompt)
            added = append_to_jsonl(records, output_file)
            total_added += added

    print(f"\n[AutoScan] Multi-File Ingestion Complete! Total new records added: {total_added}")

def import_kaggle_autoscan(handle, user_col="auto", assistant_col="auto", output_file="Finetune/training_data.jsonl", system_prompt=None):
    import kagglehub
    print(f"[AutoScan] Downloading Kaggle dataset '{handle}'...")
    folder = kagglehub.dataset_download(handle)
    scan_path_pattern(folder, user_col, assistant_col, output_file, system_prompt)

def import_hf_autoscan(repo_id, user_col="auto", assistant_col="auto", output_file="Finetune/training_data.jsonl", system_prompt=None):
    from datasets import load_dataset
    print(f"[AutoScan] Downloading Hugging Face dataset '{repo_id}'...")
    ds = load_dataset(repo_id)

    total_added = 0
    if hasattr(ds, "keys"):
        splits = list(ds.keys())
        print(f"[AutoScan] Found Hugging Face dataset splits: {splits}")
        for split in splits:
            print(f"[AutoScan] Processing split '{split}'...")
            df = ds[split].to_pandas()
            print(f"[AutoScan] Split '{split}' shape: {df.shape}")
            records = process_dataframe(df, user_col, assistant_col, system_prompt)
            added = append_to_jsonl(records, output_file)
            total_added += added
    else:
        df = ds.to_pandas()
        print(f"[AutoScan] Dataset shape: {df.shape}")
        records = process_dataframe(df, user_col, assistant_col, system_prompt)
        added = append_to_jsonl(records, output_file)
        total_added += added

    print(f"\n[AutoScan] Hugging Face Ingestion Complete! Total new records added: {total_added}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Auto-Scanning Dataset Importer for Local SLM Fine-Tuning")
    parser.add_argument("--kaggle", help="Kaggle handle (e.g. owner/dataset-name)")
    parser.add_argument("--hf", help="Hugging Face repo (e.g. sjmathy/epic_kitchen_100_resume)")
    parser.add_argument("--folder", help="Local folder path or wildcard to recursively scan and ingest all files")
    parser.add_argument("--file", help="Local file path or wildcard pattern (e.g. D:/models/Training/*.zip)")
    parser.add_argument("--user_col", default="auto", help="Column for prompt. Default 'auto' detects automatically.")
    parser.add_argument("--assistant_col", default="auto", help="Column for target response. Default 'auto' detects automatically.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt")
    parser.add_argument("--output", default="Finetune/training_data.jsonl", help="Output JSONL file")

    args = parser.parse_args()

    target_pattern = args.folder or args.file
    if args.kaggle:
        import_kaggle_autoscan(args.kaggle, args.user_col, args.assistant_col, args.output, args.system_prompt)
    elif args.hf:
        import_hf_autoscan(args.hf, args.user_col, args.assistant_col, args.output, args.system_prompt)
    elif target_pattern:
        scan_path_pattern(target_pattern, args.user_col, args.assistant_col, args.output, args.system_prompt)
    else:
        print("Usage examples:")
        print("  Ingest Hugging Face dataset: python Finetune/dataset_importer.py --hf sjmathy/epic_kitchen_100_resume")
        print("  Ingest Kaggle repo: python Finetune/dataset_importer.py --kaggle owner/repo")
        print("  Ingest all zips: python Finetune/dataset_importer.py --file 'D:/models/Training/*.zip'")
