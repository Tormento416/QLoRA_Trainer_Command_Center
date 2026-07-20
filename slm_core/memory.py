import os
import json
from typing import List, Dict, Optional

class ShortTermMemory:
    """Manages sliding context window for multi-turn conversations."""
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history)
        return messages

    def _trim(self):
        # max_turns represents number of user-assistant pairs (2 * max_turns entries)
        max_entries = self.max_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def clear(self):
        self.history.clear()


class LongTermMemory:
    """Manages persistent fact memory stored on disk (memory_bank.json)."""
    def __init__(self, storage_path: str = "data/memory_bank.json"):
        self.storage_path = storage_path
        self.memories: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception as e:
                print(f"[MemoryBank] Failed to load memory file: {e}")
                self.memories = {}
        else:
            self.memories = {}

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MemoryBank] Failed to save memory file: {e}")

    def add_fact(self, key: str, value: str):
        self.memories[key.strip().lower()] = value.strip()
        self.save()

    def remove_fact(self, key: str):
        key = key.strip().lower()
        if key in self.memories:
            del self.memories[key]
            self.save()

    def get_relevant_memories(self, query: str) -> Dict[str, str]:
        """Returns facts where the key or value overlaps with the query, or all facts if total is small."""
        if not self.memories:
            return {}
        
        query_words = set(query.lower().split())
        matched = {}
        for k, v in self.memories.items():
            # Match if key/value words appear in user prompt or if facts bank is small (< 20 items)
            if len(self.memories) <= 15:
                matched[k] = v
            else:
                combined_text = f"{k} {v}".lower()
                if any(word in combined_text for word in query_words if len(word) > 3):
                    matched[k] = v
        return matched

    def build_system_prompt(self, base_prompt: str, user_query: str) -> str:
        relevant = self.get_relevant_memories(user_query)
        if not relevant:
            return base_prompt

        memory_str = "\n".join([f"- {k}: {v}" for k, v in relevant.items()])
        return (
            f"{base_prompt}\n\n"
            f"=== MEMORY BANK (User Facts & Preferences) ===\n"
            f"{memory_str}\n"
            f"Use the above stored memory facts when relevant to answer the user accurately."
        )
