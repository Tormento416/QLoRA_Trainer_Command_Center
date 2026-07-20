# Alias wrapper for dataset_importer.py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from dataset_importer import scan_path_pattern, import_kaggle_autoscan, import_hf_autoscan, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Auto-Scanning Dataset Importer (Alias)")
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
        print("  Ingest Hugging Face dataset: python Finetune/data_importer.py --hf sjmathy/epic_kitchen_100_resume")
        print("  Ingest Kaggle repo: python Finetune/data_importer.py --kaggle owner/repo")
        print("  Ingest all zips: python Finetune/data_importer.py --file 'D:/models/Training/*.zip'")
