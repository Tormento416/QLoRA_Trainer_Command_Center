import os
import json
import kagglehub

print("[LinkedIn Skills] Downloading dataset 'maneeshdisodia/employment-skills'...")
folder_path = kagglehub.dataset_download("maneeshdisodia/employment-skills")
skills_file_path = os.path.join(folder_path, "linkedin skill")

if not os.path.exists(skills_file_path):
    # Fallback to look for file
    for root, _, files in os.walk(folder_path):
        if files:
            skills_file_path = os.path.join(root, files[0])
            break

print(f"[LinkedIn Skills] Reading skill list from '{skills_file_path}'...")
with open(skills_file_path, "r", encoding="utf-8", errors="ignore") as f:
    skills = [line.strip() for line in f if line.strip()]

print(f"[LinkedIn Skills] Found {len(skills)} unique LinkedIn skills.")

output_file = "Finetune/training_data.jsonl"
print(f"[LinkedIn Skills] Converting skills into training prompts and appending to '{output_file}'...")

records = []
chunk_size = 12

for i in range(0, len(skills), chunk_size):
    chunk = skills[i : i + chunk_size]
    skills_str = ", ".join(chunk)

    first_char = chunk[0][0].upper() if chunk[0] else "A"
    
    # Prompt 1: Skill suggestions
    user_prompt = f"List standard professional LinkedIn skills starting with or related to '{first_char}':"
    assistant_response = f"Recognized LinkedIn Skills: {skills_str}."
    
    entry = {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    records.append(entry)

os.makedirs("Finetune", exist_ok=True)
with open(output_file, "a", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[LinkedIn Skills] Successfully added {len(records)} skill-recommendation training entries to '{output_file}'!")
