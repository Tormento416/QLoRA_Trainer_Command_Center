import os
import sys
import time
import glob
import string
import json
import io
import threading
import multiprocessing
import customtkinter as ctk

# Ensure PyInstaller freeze support at top level
multiprocessing.freeze_support()

# Set CustomTkinter Theme to Dark Mode with Blue/Cyan Accents
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Import Torch & Transformers
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Global inference model cache
_CACHED_MODEL = None
_CACHED_TOKENIZER = None
_CACHED_MODEL_PATH = None
_CACHED_ADAPTER_PATH = None


def resolve_workspace_path(rel_path):
    """Resolves relative file paths to the workspace root across PyInstaller frozen builds."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        # 1. Check parent workspace directories
        candidate_1 = os.path.abspath(os.path.join(exe_dir, "..", "..", rel_path))
        if os.path.exists(candidate_1) or os.path.exists(os.path.dirname(candidate_1)):
            return candidate_1
        # 2. Check executable directory
        candidate_2 = os.path.abspath(os.path.join(exe_dir, rel_path))
        if os.path.exists(candidate_2) or os.path.exists(os.path.dirname(candidate_2)):
            return candidate_2
    return os.path.abspath(rel_path)


class DriveScanner:
    """Scans local PC drives for small language model (SLM) directory structures."""

    @staticmethod
    def get_windows_drives():
        drives = []
        if sys.platform == "win32":
            for letter in string.ascii_uppercase:
                drive_path = f"{letter}:\\"
                if os.path.exists(drive_path):
                    drives.append(drive_path)
        else:
            drives = ["/"]
        return drives

    @classmethod
    def scan_for_slm_models(cls, callback_progress=None):
        discovered_models = []
        drives = cls.get_windows_drives()
        
        priority_paths = [
            r"D:\models",
            r"C:\models",
            os.path.expanduser("~\\.cache\\huggingface\\hub"),
        ]

        signature_files = ["config.json", "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]

        for path in priority_paths:
            if os.path.exists(path):
                if callback_progress:
                    callback_progress(f"Scanning priority location: {path}...")
                cls._check_dir_recursive(path, signature_files, discovered_models, max_depth=3)

        for drive in drives:
            if callback_progress:
                callback_progress(f"Scanning drive {drive}...")
            cls._check_dir_recursive(drive, signature_files, discovered_models, max_depth=2)

        # Filter out assistant draft subfolders unless no base models found
        filtered = [m for m in discovered_models if not os.path.basename(m).lower().endswith(("-assistant", "-assistant-v1"))]
        final_list = filtered if filtered else discovered_models

        # Priority sort: D:\models first
        unique_models = sorted(list(set(final_list)), key=lambda x: (0 if os.path.abspath(x) == os.path.abspath(r"D:\models") else 1, x))
        return unique_models

    @classmethod
    def _check_dir_recursive(cls, current_dir, signatures, results, max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return
        try:
            entries = os.listdir(current_dir)
        except (PermissionError, OSError):
            return

        has_config = "config.json" in entries
        has_weights = any(sig in entries for sig in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"])

        folder_name = os.path.basename(current_dir).lower()
        if has_config and has_weights and not folder_name.endswith(("-assistant", "-assistant-v1")):
            results.append(os.path.abspath(current_dir))

        skip_folders = {"$recycle.bin", "system volume information", "windows", "program files", "program files (x86)", ".git", "node_modules", "__pycache__"}
        for entry in entries:
            if entry.lower() in skip_folders or entry.startswith("."):
                continue
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path) and not os.path.islink(full_path):
                cls._check_dir_recursive(full_path, signatures, results, max_depth, current_depth + 1)


class VRAMOptimizer:
    """Calculates recommended QLoRA fine-tuning hyperparameters based on Target VRAM allocation."""

    @staticmethod
    def get_available_vram_gb():
        if HAS_TORCH and torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return round(total_bytes / (1024 ** 3), 1)
        return 12.0

    @staticmethod
    def get_safe_vram_limit():
        total = VRAMOptimizer.get_available_vram_gb()
        # Automatically lower to 10% below max limit (90% of total VRAM) for safe overhead
        safe_vram = round(total * 0.90, 1)
        return safe_vram

    @classmethod
    def calculate_recommended_settings(cls, target_vram_gb: float):
        vram = float(target_vram_gb)
        
        if vram <= 4.5:
            return {
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "effective_batch_size": 16,
                "max_length": 256,
                "max_steps": 3000,
                "learning_rate": "2e-4",
                "sub_sample": 15000,
                "quantization": "4-bit NF4",
                "estimated_vram": "~3.8 - 4.2 GB",
                "profile_name": "Low Memory (4GB VRAM)"
            }
        elif vram <= 6.5:
            return {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 16,
                "max_length": 256,
                "max_steps": 4500,
                "learning_rate": "2e-4",
                "sub_sample": 25000,
                "quantization": "4-bit NF4",
                "estimated_vram": "~5.2 - 6.0 GB",
                "profile_name": "Standard (6GB VRAM)"
            }
        elif vram <= 9.5:
            return {
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "effective_batch_size": 16,
                "max_length": 384,
                "max_steps": 9000,
                "learning_rate": "2e-4",
                "sub_sample": 45000,
                "quantization": "4-bit NF4",
                "estimated_vram": "~7.8 - 8.4 GB",
                "profile_name": "Optimized 9GB Cap (100% Stable)"
            }
        elif vram <= 11.5:
            # 10.8 GB Safe Cap Profile for RTX 4070 SUPER (12GB)
            return {
                "batch_size": 6,
                "gradient_accumulation_steps": 3,
                "effective_batch_size": 18,
                "max_length": 384,
                "max_steps": 18000,
                "learning_rate": "2e-4",
                "sub_sample": 0,
                "quantization": "4-bit NF4",
                "estimated_vram": "~9.5 - 10.8 GB",
                "profile_name": "High Performance (12GB GPU Cap)"
            }
        elif vram <= 16.5:
            # 16GB GPUs (RTX 4080 / 16GB VRAM)
            return {
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "effective_batch_size": 16,
                "max_length": 512,
                "max_steps": 24000,
                "learning_rate": "2e-4",
                "sub_sample": 0,
                "quantization": "4-bit NF4",
                "estimated_vram": "~12.5 - 14.8 GB",
                "profile_name": "Ultra Performance (16GB VRAM)"
            }
        else:
            # 24GB+ Enterprise GPUs (RTX 4090 / A100 / H100)
            return {
                "batch_size": 12,
                "gradient_accumulation_steps": 2,
                "effective_batch_size": 24,
                "max_length": 512,
                "max_steps": 30000,
                "learning_rate": "2e-4",
                "sub_sample": 0,
                "quantization": "4-bit NF4",
                "estimated_vram": "~18.0 - 22.5 GB",
                "profile_name": "Enterprise Max (24GB+ VRAM)"
            }


class OutputRedirector(io.TextIOBase):
    """Redirects stdout/stderr to a CustomTkinter text widget in real-time."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, s):
        def _append():
            self.text_widget.insert("end", s)
            self.text_widget.see("end")
        if s:
            self.text_widget.after(0, _append)
        return len(s)

    def flush(self):
        pass


class SettingsManager:
    SETTINGS_FILE = 'user_settings.json'
    
    @classmethod
    def load(cls):
        try:
            with open(cls.SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
            
    @classmethod
    def save(cls, data):
        settings = cls.load()
        settings.update(data)
        with open(cls.SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)

class ModernRezSLMApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Generic QLoRA Trainer")
        self.geometry("980x740")
        self.minsize(900, 660)

        # Application State
        self.discovered_models = []
        self.selected_model_path = ""
        self.is_training = False
        self.stop_requested = False

        self._build_header()

        # Dynamic Content Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=10)

        self.show_step_1_start()

        # Footer
        footer = ctk.CTkLabel(self, text="(Built by A. Sousa/Tormento416)", font=ctk.CTkFont(family="Inter", size=10), text_color="#64748B")
        footer.pack(side="bottom", pady=5)

    def _build_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="#111827", corner_radius=12)
        header_frame.pack(fill="x", side="top", padx=20, pady=(16, 4))

        title_lbl = ctk.CTkLabel(header_frame, text="⚡ Generic QLoRA Trainer", font=ctk.CTkFont(family="Inter", size=18, weight="bold"), text_color="#06B6D4")
        title_lbl.pack(side="left", padx=16, pady=12)

        self.gpu_lbl = ctk.CTkLabel(header_frame, text="GPU: Detecting...", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#EAB308")
        self.gpu_lbl.pack(side="right", padx=16, pady=12)

        if HAS_TORCH:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = VRAMOptimizer.get_available_vram_gb()
                safe_vram = VRAMOptimizer.get_safe_vram_limit()
                self.gpu_lbl.configure(text=f"🎮 {gpu_name} ({total_vram} GB Max | {safe_vram} GB Safe Cap)")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.gpu_lbl.configure(text="🎮 Apple Silicon (MPS Enabled)")
            else:
                self.gpu_lbl.configure(text="💻 CPU Only Mode (Not Recommended)")

    def _clear_main_container(self):
        for widget in self.main_container.winfo_children():
            widget.destroy()

    # ---------------------------------------------------------
    # STEP 1: WELCOME & DRIVE SCAN SCREEN
    # ---------------------------------------------------------
    def show_step_1_start(self):
        self._clear_main_container()

        card = ctk.CTkFrame(self.main_container, fg_color="#111827", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, pady=10)

        welcome_title = ctk.CTkLabel(card, text="Welcome to Generic QLoRA Trainer", font=ctk.CTkFont(family="Inter", size=22, weight="bold"), text_color="#F8FAFC")
        welcome_title.pack(pady=(40, 10))

        welcome_desc = ctk.CTkLabel(
            card,
            text="Scan your PC and attached drives to discover local SLM/LLM models,\nthen launch high-performance generic QLoRA fine-tuning.",
            font=ctk.CTkFont(family="Inter", size=13),
            text_color="#94A3B8",
            justify="center"
        )
        welcome_desc.pack(pady=(0, 36))

        self.scan_action_btn = ctk.CTkButton(
            card,
            text="🔍  Scan PC & Drives for SLM Models",
            font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            text_color="#FFFFFF",
            corner_radius=10,
            height=44,
            width=320,
            command=self.execute_drive_scan_async
        )
        self.scan_action_btn.pack(pady=10)

        self.scan_status_lbl = ctk.CTkLabel(card, text="Click above to scan for local model directories...", font=ctk.CTkFont(family="Consolas", size=11), text_color="#06B6D4")
        self.scan_status_lbl.pack(pady=(20, 0))

    def execute_drive_scan_async(self):
        self.scan_action_btn.configure(state="disabled", text="⏳  Scanning PC Drives...", fg_color="#334155")
        threading.Thread(target=self._scan_thread_func, daemon=True).start()

    def _scan_thread_func(self):
        models = DriveScanner.scan_for_slm_models(
            callback_progress=lambda msg: self.after(0, self._update_scan_progress, msg)
        )
        self.after(0, self._on_scan_completed, models)

    def _update_scan_progress(self, msg):
        self.scan_status_lbl.configure(text=msg)

    def _on_scan_completed(self, models):
        self.discovered_models = models
        settings = SettingsManager.load()
        saved_model = settings.get("model_path", "")
        if saved_model and saved_model not in self.discovered_models:
            self.discovered_models.insert(0, saved_model)
        if not self.discovered_models:
            self.discovered_models = [""]
        
        self.show_step_2_select_model()

    # ---------------------------------------------------------
    # STEP 2: MODEL SELECTION & WORKFLOW CHOICE
    # ---------------------------------------------------------
    def show_step_2_select_model(self):
        self._clear_main_container()

        card = ctk.CTkFrame(self.main_container, fg_color="#111827", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, pady=10, padx=10)

        header_lbl = ctk.CTkLabel(card, text="Step 1: Select Model & Workflow Task", font=ctk.CTkFont(family="Inter", size=16, weight="bold"), text_color="#F8FAFC")
        header_lbl.pack(anchor="w", padx=24, pady=(24, 16))

        # Model Dropdown Card
        model_card = ctk.CTkFrame(card, fg_color="#0B0F19", corner_radius=12, border_width=1, border_color="#1E293B")
        model_card.pack(fill="x", padx=24, pady=(0, 24))

        if len(self.discovered_models) > 1:
            lbl_text = f"Discovered {len(self.discovered_models)} SLM model directories on your PC:"
        else:
            lbl_text = "Discovered 1 local SLM model directory:"

        tk_lbl = ctk.CTkLabel(model_card, text=lbl_text, font=ctk.CTkFont(family="Inter", size=12, weight="bold"), text_color="#06B6D4")
        tk_lbl.pack(anchor="w", padx=16, pady=(14, 6))

        self.model_combo_var = ctk.StringVar(value=self.discovered_models[0])
        self.model_combo = ctk.CTkOptionMenu(
            model_card,
            variable=self.model_combo_var,
            values=self.discovered_models,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1E293B",
            button_color="#0284C7",
            button_hover_color="#0369A1",
            dropdown_font=ctk.CTkFont(family="Consolas", size=11),
            height=36
        )
        self.model_combo.pack(fill="x", padx=16, pady=(0, 10))

        # Browse custom folder
        b_row = ctk.CTkFrame(model_card, fg_color="transparent")
        b_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(b_row, text="Or select a custom model directory:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").pack(side="left")
        ctk.CTkButton(b_row, text="📁 Browse Custom Folder...", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), fg_color="#334155", hover_color="#475569", width=180, height=28, command=self.browse_custom_model).pack(side="right")

        # Workflow Choice Section
        wf_card = ctk.CTkFrame(card, fg_color="#0B0F19", corner_radius=12, border_width=1, border_color="#1E293B")
        wf_card.pack(fill="x", padx=24, pady=8)

        ctk.CTkLabel(wf_card, text="Select Mode to Launch:", font=ctk.CTkFont(family="Inter", size=13, weight="bold"), text_color="#EAB308").pack(anchor="w", padx=20, pady=(16, 14))

        btn_row = ctk.CTkFrame(wf_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 20))

        tr_btn = ctk.CTkButton(
            btn_row,
            text="🔥  QLoRA Fine-Tuning",
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            text_color="#FFFFFF",
            corner_radius=10,
            height=44,
            command=self.show_step_3b_training
        )
        tr_btn.pack(side="left", expand=True, fill="x")

    def browse_custom_model(self):
        path = ctk.filedialog.askdirectory(title="Select Custom SLM Model Directory")
        if path:
            abs_p = os.path.abspath(path)
            if abs_p not in self.discovered_models:
                self.discovered_models.insert(0, abs_p)
                self.model_combo.configure(values=self.discovered_models)
            self.model_combo_var.set(abs_p)

    # ---------------------------------------------------------
    # STEP 3A: INFERENCE MODE PANEL
    # ---------------------------------------------------------
    def show_step_3a_inference(self):
        self.selected_model_path = self.model_combo_var.get().strip()
        self._clear_main_container()

        card = ctk.CTkFrame(self.main_container, fg_color="#111827", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, pady=4, padx=4)

        # Header Row
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(16, 8))

        ctk.CTkLabel(top_row, text=f"💬 Inference Mode — Model: {os.path.basename(self.selected_model_path)}", font=ctk.CTkFont(family="Inter", size=14, weight="bold"), text_color="#10B981").pack(side="left")
        ctk.CTkButton(top_row, text="← Change Model", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), fg_color="#334155", hover_color="#475569", width=120, height=28, command=self.show_step_2_select_model).pack(side="right")

        # Configuration Row
        cfg_row = ctk.CTkFrame(card, fg_color="#0B0F19", corner_radius=10)
        cfg_row.pack(fill="x", padx=20, pady=6)

        ctk.CTkLabel(cfg_row, text="Max VRAM Allocated (GB):", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").pack(side="left", padx=12, pady=8)
        
        # Set default to 10% below max limit (90% safe cap)
        safe_vram = VRAMOptimizer.get_safe_vram_limit()
        self.inf_vram_var = ctk.StringVar(value=str(safe_vram))
        self.inf_vram_entry = ctk.CTkEntry(cfg_row, textvariable=self.inf_vram_var, width=60, font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), fg_color="#040711", text_color="#EAB308")
        self.inf_vram_entry.pack(side="left", padx=(0, 20), pady=8)

        ctk.CTkLabel(cfg_row, text="Max Tokens:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").pack(side="left", padx=(0, 4), pady=8)
        self.inf_max_tokens_var = ctk.StringVar(value="256")
        self.inf_tokens_entry = ctk.CTkEntry(cfg_row, textvariable=self.inf_max_tokens_var, width=64, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711")
        self.inf_tokens_entry.pack(side="left", padx=(0, 20), pady=8)

        ctk.CTkLabel(cfg_row, text="Temperature:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").pack(side="left", padx=(0, 4), pady=8)
        self.inf_temp_var = ctk.StringVar(value="0.2")
        self.inf_temp_entry = ctk.CTkEntry(cfg_row, textvariable=self.inf_temp_var, width=54, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711")
        self.inf_temp_entry.pack(side="left", pady=8)

        # Adapter Row
        ad_row = ctk.CTkFrame(card, fg_color="transparent")
        ad_row.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(ad_row, text="Optional LoRA Adapter Path:", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").pack(side="left")
        self.inf_adapter_entry = ctk.CTkEntry(ad_row, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", text_color="#A5F3FC")
        self.inf_adapter_entry.insert(0, resolve_workspace_path("Finetune/output_adapter"))
        self.inf_adapter_entry.pack(side="left", fill="x", expand=True, padx=8)

        # Pro Tip Banner: LinkedIn Profile Source of Truth
        tip_frame = ctk.CTkFrame(card, fg_color="#062038", corner_radius=8, border_width=1, border_color="#0284C7")
        tip_frame.pack(fill="x", padx=20, pady=4)

        tip_txt = "💡 Pro Tip: Attach your LinkedIn Profile PDF as your source of truth for resume generation!\n(In LinkedIn: Go to Profile > Resources > Download PDF)"
        ctk.CTkLabel(tip_frame, text=tip_txt, font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#38BDF8", justify="left").pack(anchor="w", padx=12, pady=6)

        # File Attachment Row (Resume / JD / TXT / Image / PDF)
        prompt_header_row = ctk.CTkFrame(card, fg_color="transparent")
        prompt_header_row.pack(fill="x", padx=20, pady=(8, 2))

        ctk.CTkLabel(prompt_header_row, text="Inference Prompt:", font=ctk.CTkFont(family="Inter", size=12, weight="bold"), text_color="#F8FAFC").pack(side="left")
        ctk.CTkButton(
            prompt_header_row,
            text="📄  Attach File (Resume / JD / TXT / PDF)...",
            font=ctk.CTkFont(family="Inter", size=11, weight="bold"),
            fg_color="#0284C7",
            hover_color="#0369A1",
            width=220,
            height=26,
            command=self.attach_file_to_prompt
        ).pack(side="right")

        # Prompt Input
        self.inf_prompt_text = ctk.CTkTextbox(card, height=140, font=ctk.CTkFont(family="Inter", size=12), fg_color="#040711", text_color="#FFFFFF")
        self.inf_prompt_text.insert("1.0", "Analyze candidate resume summary: Senior Go developer with PyTorch and CUDA experience.")
        self.inf_prompt_text.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        # Run Button
        self.run_inf_btn = ctk.CTkButton(
            card,
            text="⚡  Run Local SLM Inference",
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            height=40,
            command=self.execute_inference_in_process
        )
        self.run_inf_btn.pack(fill="x", padx=20, pady=4)

        # Output Box
        ctk.CTkLabel(card, text="Response Output:", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(8, 2))
        self.inf_output_text = ctk.CTkTextbox(card, height=220, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#02040A", text_color="#67E8F9")
        self.inf_output_text.pack(fill="both", expand=True, padx=20, pady=(0, 16))

    def attach_file_to_prompt(self):
        file_path = ctk.filedialog.askopenfilename(
            title="Select Attachment File (Resume / JD / TXT / PDF / Image)",
            filetypes=[("Text / Document / Image Files", "*.txt *.pdf *.md *.json *.jsonl *.csv *.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if file_path:
            filename = os.path.basename(file_path)
            content_str = ""
            try:
                if file_path.lower().endswith((".txt", ".md", ".json", ".jsonl", ".csv")):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content_str = f.read()
                elif file_path.lower().endswith(".pdf"):
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(file_path)
                        pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                        content_str = "\n".join(pages_text)
                    except Exception:
                        content_str = f"[Attached PDF File: {file_path}]"
                else:
                    content_str = f"[Attached File Reference: {file_path}]"
            except Exception as e:
                content_str = f"[File Attach Error: {e}]"

            attachment_text = f"\n\n--- ATTACHED FILE CONTEXT ({filename}) ---\n{content_str}\n--- END ATTACHMENT ---\n"
            self.inf_prompt_text.insert("end", attachment_text)
            self.inf_prompt_text.see("end")

    def execute_inference_in_process(self):
        model_path = self.selected_model_path
        adapter_path = self.inf_adapter_entry.get().strip()
        prompt = self.inf_prompt_text.get("1.0", "end").strip()
        try:
            max_tokens = int(self.inf_max_tokens_var.get())
        except ValueError:
            max_tokens = 256

        self.run_inf_btn.configure(state="disabled", text="⏳  Loading Model & Generating Response...", fg_color="#334155")
        self.inf_output_text.delete("1.0", "end")
        self.inf_output_text.insert("1.0", "[SLM Engine] Loading model weights into GPU VRAM...\n")

        threading.Thread(target=self._inference_thread_func, args=(model_path, adapter_path, prompt, max_tokens), daemon=True).start()

    def _inference_thread_func(self, model_path, adapter_path, prompt, max_tokens):
        global _CACHED_MODEL, _CACHED_TOKENIZER, _CACHED_MODEL_PATH, _CACHED_ADAPTER_PATH
        output_text = ""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path or _CACHED_ADAPTER_PATH != adapter_path:
                print(f"[SLM] Loading base model from {model_path}...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"": 0})

                if os.path.exists(adapter_path):
                    try:
                        from peft import PeftModel
                        model = PeftModel.from_pretrained(model, adapter_path)
                        print("[SLM] LoRA adapter attached successfully.")
                    except Exception as ea:
                        print(f"[SLM] LoRA adapter skip: {ea}")

                _CACHED_MODEL = model
                _CACHED_TOKENIZER = tokenizer
                _CACHED_MODEL_PATH = model_path
                _CACHED_ADAPTER_PATH = adapter_path
            else:
                model = _CACHED_MODEL
                tokenizer = _CACHED_TOKENIZER

            if getattr(tokenizer, "chat_template", None) is not None:
                try:
                    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                except Exception:
                    formatted_prompt = f"User: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"

            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        except Exception as e:
            output_text = f"Inference Exception: {e}"

        self.after(0, self._on_inference_completed, output_text)

    def _on_inference_completed(self, result_text):
        self.run_inf_btn.configure(state="normal", text="⚡  Run Local SLM Inference", fg_color="#10B981")
        self.inf_output_text.delete("1.0", "end")
        self.inf_output_text.insert("1.0", result_text)

    # ---------------------------------------------------------
    # STEP 3B: TRAINING MODE PANEL
    # ---------------------------------------------------------
    def show_step_3b_training(self):
        self.selected_model_path = self.model_combo_var.get().strip()
        self._clear_main_container()

        card = ctk.CTkFrame(self.main_container, fg_color="#111827", corner_radius=14, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, pady=4, padx=4)

        # 1. Header Row
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=16, pady=(12, 4))

        ctk.CTkLabel(top_row, text=f"🔥 QLoRA Fine-Tuning — Base Model: {os.path.basename(self.selected_model_path)}", font=ctk.CTkFont(family="Inter", size=14, weight="bold"), text_color="#EF4444").pack(side="left")
        ctk.CTkButton(top_row, text="← Change Model", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), fg_color="#334155", hover_color="#475569", width=120, height=28, command=self.show_step_2_select_model).pack(side="right")

        # 2. Dataset & Output Folder Paths
        d_frame = ctk.CTkFrame(card, fg_color="#0B0F19", corner_radius=10)
        d_frame.pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(d_frame, text="Dataset File:", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").grid(row=0, column=0, padx=10, pady=4, sticky="w")
        self.tr_dataset_entry = ctk.CTkEntry(d_frame, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=28)
        
        default_ds = resolve_workspace_path("Finetune/training_data.jsonl")
        if not os.path.exists(default_ds):
            alt_ds = os.path.abspath("Finetune/training_data.jsonl")
            if os.path.exists(alt_ds):
                default_ds = alt_ds
        self.tr_dataset_entry.insert(0, default_ds)
        self.tr_dataset_entry.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        ctk.CTkButton(d_frame, text="Browse...", font=ctk.CTkFont(family="Inter", size=11), fg_color="#334155", width=70, height=26, command=self.browse_tr_dataset).grid(row=0, column=2, padx=10, pady=4)

        ctk.CTkLabel(d_frame, text="Output Directory:", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        self.tr_out_entry = ctk.CTkEntry(d_frame, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=28)
        self.tr_out_entry.insert(0, resolve_workspace_path("Finetune/output_adapter"))
        self.tr_out_entry.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        ctk.CTkButton(d_frame, text="Browse...", font=ctk.CTkFont(family="Inter", size=11), fg_color="#334155", width=70, height=26, command=self.browse_tr_out_dir).grid(row=1, column=2, padx=10, pady=4)

        d_frame.columnconfigure(1, weight=1)

        # 3. Settings Card (VRAM & Duration)
        mode_card = ctk.CTkFrame(card, fg_color="#0B0F19", corner_radius=10)
        mode_card.pack(fill="x", padx=16, pady=4)

        self.tr_out_var = ctk.StringVar(value=SettingsManager.load().get("output_dir", "Finetune/output_adapter"))

        self.tr_mode_var = ctk.StringVar(value="auto")

        rb_auto = ctk.CTkRadioButton(
            mode_card,
            text="🎯 Max VRAM Allocated (Auto-Optimize Settings)",
            variable=self.tr_mode_var,
            value="auto",
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            text_color="#EAB308",
            command=self.toggle_tr_mode_inputs
        )
        rb_auto.pack(anchor="w", padx=14, pady=(6, 2))

        v_row = ctk.CTkFrame(mode_card, fg_color="transparent")
        v_row.pack(fill="x", padx=28, pady=(0, 2))

        ctk.CTkLabel(v_row, text="Target Max VRAM (GB):", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").pack(side="left")
        
        safe_vram_cap = VRAMOptimizer.get_safe_vram_limit()
        self.tr_vram_var = ctk.StringVar(value=str(safe_vram_cap))
        self.tr_vram_entry = ctk.CTkEntry(v_row, textvariable=self.tr_vram_var, width=54, font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), fg_color="#040711", text_color="#EAB308", height=26)
        self.tr_vram_entry.pack(side="left", padx=6)
        self.tr_vram_var.trace_add("write", self.update_vram_summary)

        ctk.CTkButton(v_row, text="⚡ Apply VRAM Settings", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), fg_color="#EAB308", hover_color="#CA8A04", text_color="#000000", width=140, height=26, command=self.apply_tr_vram_rec).pack(side="left", padx=6)

        self.vram_summary_lbl = ctk.CTkLabel(v_row, text="Rec: Batch 6 | Grad Accum 3 (Eff Batch 18) | Steps 18,000", font=ctk.CTkFont(family="Inter", size=10), text_color="#10B981")
        self.vram_summary_lbl.pack(side="left", padx=8)

        # Timed Training Setting
        time_row = ctk.CTkFrame(mode_card, fg_color="transparent")
        time_row.pack(fill="x", padx=28, pady=(0, 2))

        ctk.CTkLabel(time_row, text="⏱️ Max Training Duration (Minutes, 0=Unlimited):", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").pack(side="left")
        self.tr_time_limit_var = ctk.StringVar(value="0")
        self.tr_time_entry = ctk.CTkEntry(time_row, textvariable=self.tr_time_limit_var, width=54, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=26)
        self.tr_time_entry.pack(side="left", padx=6)

        rb_custom = ctk.CTkRadioButton(
            mode_card,
            text="⚙️ Custom Settings (Manual Hyperparameters)",
            variable=self.tr_mode_var,
            value="custom",
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
            text_color="#F8FAFC",
            command=self.toggle_tr_mode_inputs
        )
        rb_custom.pack(anchor="w", padx=14, pady=(2, 2))

        # Custom inputs grid
        self.custom_inputs_frame = ctk.CTkFrame(mode_card, fg_color="transparent")
        self.custom_inputs_frame.pack(fill="x", padx=28, pady=(0, 4))

        ctk.CTkLabel(self.custom_inputs_frame, text="Batch Size:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").grid(row=0, column=0, sticky="w")
        self.tr_batch_var = ctk.StringVar(value="6")
        self.tr_batch_entry = ctk.CTkEntry(self.custom_inputs_frame, textvariable=self.tr_batch_var, width=50, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=26)
        self.tr_batch_entry.grid(row=0, column=1, padx=(4, 12), sticky="w")

        ctk.CTkLabel(self.custom_inputs_frame, text="Grad Accum:", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#F8FAFC").grid(row=0, column=2, sticky="w")
        self.tr_grad_accum_var = ctk.StringVar(value="8")
        self.tr_grad_accum_combo = ctk.CTkOptionMenu(
            self.custom_inputs_frame,
            variable=self.tr_grad_accum_var,
            values=["Auto (0)", "1", "2", "4", "8", "16", "32"],
            width=90, height=26,
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            fg_color="#1E293B", button_color="#0284C7",
            command=lambda choice: self.update_vram_summary()
        )
        self.tr_grad_accum_combo.grid(row=0, column=3, padx=(4, 12), sticky="w")

        ctk.CTkLabel(self.custom_inputs_frame, text="Learning Rate:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").grid(row=0, column=4, sticky="w")
        self.tr_lr_var = ctk.StringVar(value="2e-4")
        self.tr_lr_entry = ctk.CTkEntry(self.custom_inputs_frame, textvariable=self.tr_lr_var, width=65, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=26)
        self.tr_lr_entry.grid(row=0, column=5, padx=(4, 12), sticky="w")

        ctk.CTkLabel(self.custom_inputs_frame, text="Max Steps:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").grid(row=0, column=6, sticky="w")
        self.tr_steps_var = ctk.StringVar(value="18000")
        self.tr_steps_entry = ctk.CTkEntry(self.custom_inputs_frame, textvariable=self.tr_steps_var, width=65, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#040711", height=26)
        self.tr_steps_entry.grid(row=0, column=7, padx=(4, 12), sticky="w")

        self.tr_steps_var.trace_add("write", self.update_vram_summary)
        self.tr_grad_accum_var.trace_add("write", self.update_vram_summary)

        # Advanced Settings Row
        self.adv_inputs_frame = ctk.CTkFrame(mode_card, fg_color="transparent")
        self.adv_inputs_frame.pack(fill="x", padx=28, pady=(0, 8))

        ctk.CTkLabel(self.adv_inputs_frame, text="Mixed Precision:", font=ctk.CTkFont(family="Inter", size=11), text_color="#94A3B8").grid(row=0, column=0, sticky="w")
        self.tr_precision_var = ctk.StringVar(value="Auto")
        self.tr_precision_combo = ctk.CTkOptionMenu(
            self.adv_inputs_frame, 
            variable=self.tr_precision_var, 
            values=["Auto", "BF16", "FP16"],
            width=70, height=26,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#1E293B", button_color="#0284C7"
        )
        self.tr_precision_combo.grid(row=0, column=1, padx=(4, 12), sticky="w")

        self.tr_grad_ckpt_var = ctk.StringVar(value="on")
        self.tr_grad_ckpt_switch = ctk.CTkSwitch(
            self.adv_inputs_frame, 
            text="Grad Checkpointing", 
            variable=self.tr_grad_ckpt_var, 
            onvalue="on", offvalue="off",
            font=ctk.CTkFont(family="Inter", size=11),
            text_color="#94A3B8"
        )
        self.tr_grad_ckpt_switch.grid(row=0, column=2, padx=(4, 12), sticky="w")

        self.apply_tr_vram_rec()
        self.toggle_tr_mode_inputs()

        # 4. PRIMARY CONTROLS & DUAL PROGRESS BARS (RUN % & 4.94M DATASET %)
        ctrl_card = ctk.CTkFrame(card, fg_color="#0F172A", corner_radius=10, border_width=1, border_color="#334155")
        ctrl_card.pack(fill="x", padx=16, pady=4)

        tr_btn_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        tr_btn_row.pack(fill="x", padx=14, pady=(8, 6))

        self.start_tr_btn = ctk.CTkButton(
            tr_btn_row,
            text="🔥  Start QLoRA Fine-Tuning Process",
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            height=38,
            command=self.execute_training_in_process
        )
        self.start_tr_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))

        self.stop_tr_btn = ctk.CTkButton(
            tr_btn_row,
            text="🛑  STOP TRAINING",
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            fg_color="#DC2626",
            hover_color="#B91C1C",
            state="disabled",
            height=38,
            width=170,
            command=self.request_stop_training
        )
        self.stop_tr_btn.pack(side="right")

        # Live Progress Bars Frame
        progress_grid = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        progress_grid.pack(fill="x", padx=14, pady=(2, 6))

        # 4A. Run Progress (Steps / Max Steps)
        r_hdr = ctk.CTkFrame(progress_grid, fg_color="transparent")
        r_hdr.pack(fill="x")
        ctk.CTkLabel(r_hdr, text="⚡ Current Run Progress (Max Steps):", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#10B981").pack(side="left")
        self.run_pct_lbl = ctk.CTkLabel(r_hdr, text="0.0% (Step 0 / 18,000)", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#10B981")
        self.run_pct_lbl.pack(side="right")

        self.tr_progress_bar = ctk.CTkProgressBar(progress_grid, height=12, fg_color="#1E293B", progress_color="#10B981")
        self.tr_progress_bar.set(0.0)
        self.tr_progress_bar.pack(fill="x", pady=(2, 4))

        # 4B. Total Dataset Progress (of 4.94M Records)
        ds_hdr = ctk.CTkFrame(progress_grid, fg_color="transparent")
        ds_hdr.pack(fill="x")
        ctk.CTkLabel(ds_hdr, text="📊 Total Dataset Coverage (of 4.94M Records):", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#06B6D4").pack(side="left")
        self.dataset_pct_lbl = ctk.CTkLabel(ds_hdr, text="0.000% (0 / 4,940,000 records)", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#06B6D4")
        self.dataset_pct_lbl.pack(side="right")

        self.tr_dataset_progress_bar = ctk.CTkProgressBar(progress_grid, height=12, fg_color="#1E293B", progress_color="#06B6D4")
        self.tr_dataset_progress_bar.set(0.0)
        self.tr_dataset_progress_bar.pack(fill="x", pady=(2, 4))

        self.tr_progress_lbl = ctk.CTkLabel(ctrl_card, text="Status: Ready to start fine-tuning...", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#94A3B8")
        self.tr_progress_lbl.pack(anchor="w", padx=14, pady=(0, 6))

        # 5. Live Terminal Log Output Box (DYNAMICALLY SCALES WITH WINDOW RESIZE!)
        ctk.CTkLabel(card, text="Training Log Terminal (Real-time stdout — Viewbox dynamically scales):", font=ctk.CTkFont(family="Inter", size=11, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=16, pady=(4, 2))
        self.tr_log_text = ctk.CTkTextbox(card, font=ctk.CTkFont(family="Consolas", size=10), fg_color="#02040A", text_color="#4ADE80", corner_radius=8)
        self.tr_log_text.pack(fill="both", expand=True, padx=16, pady=(0, 10))

    def toggle_tr_mode_inputs(self):
        # Keep hyperparameter fields accessible so users can customize batch size even with VRAM cap active
        self.tr_batch_entry.configure(state="normal")
        self.tr_lr_entry.configure(state="normal")
        self.tr_steps_entry.configure(state="normal")
        if hasattr(self, 'tr_grad_accum_entry'):
            self.tr_grad_accum_entry.configure(state="normal")

    def update_vram_summary(self, *args):
        try:
            vram = float(self.tr_vram_var.get())
        except ValueError:
            vram = 10.8

        try:
            user_batch = int(self.tr_batch_var.get())
        except ValueError:
            user_batch = 4

        try:
            user_accum = int(self.tr_grad_accum_var.get()) if hasattr(self, 'tr_grad_accum_var') else 0
        except ValueError:
            user_accum = 0

        rec = VRAMOptimizer.calculate_recommended_settings(vram)
        grad_accum = user_accum if user_accum > 0 else (3 if user_batch == 6 else (4 if user_batch <= 4 else 2))
        eff_batch = user_batch * grad_accum

        is_override = user_batch != rec["batch_size"] or (user_accum > 0 and user_accum != (3 if rec["batch_size"] == 6 else (4 if rec["batch_size"] <= 4 else 2)))
        color_val = "#F59E0B" if is_override else "#10B981"
        tag = f"Aggressive Override (Batch {user_batch})" if is_override else rec["profile_name"]

        try:
            steps_val = int(self.tr_steps_var.get())
            steps_str = f"{steps_val:,}"
        except ValueError:
            steps_str = self.tr_steps_var.get()

        self.vram_summary_lbl.configure(
            text=f"Batch {user_batch} | Grad Accum {grad_accum} (Eff Batch {eff_batch}) | Context {rec['max_length']} | Steps {steps_str} [{tag}]",
            text_color=color_val
        )

    def apply_tr_vram_rec(self):
        try:
            vram = float(self.tr_vram_var.get())
        except ValueError:
            vram = 10.8

        rec = VRAMOptimizer.calculate_recommended_settings(vram)
        self.tr_batch_var.set(str(rec["batch_size"]))
        self.tr_steps_var.set(str(rec["max_steps"]))
        self.update_vram_summary()

    def browse_tr_dataset(self):
        p = ctk.filedialog.askopenfilename(title="Select Training Dataset File", filetypes=[("JSONL / Dataset Files", "*.jsonl *.csv *.json *.zip"), ("All Files", "*.*")])
        if p:
            self.tr_dataset_entry.delete(0, "end")
            self.tr_dataset_entry.insert(0, os.path.abspath(p))

    def browse_tr_out_dir(self):
        p = ctk.filedialog.askdirectory(title="Select Output Adapter Folder")
        if p:
            self.tr_out_entry.delete(0, "end")
            self.tr_out_entry.insert(0, os.path.abspath(p))

    def request_stop_training(self):
        if self.is_training:
            self.stop_requested = True
            self.stop_tr_btn.configure(state="disabled", text="⏳ Stopping...", fg_color="#64748B")
            self.tr_progress_lbl.configure(text="Stopping training gracefully...", text_color="#EF4444")

    def execute_training_in_process(self):
        SettingsManager.save({
            "model_path": self.selected_model_path,
            "dataset_path": self.tr_dataset_var.get().strip(),
            "output_dir": self.tr_out_var.get().strip()
        })
        dataset_path = self.tr_dataset_entry.get().strip()
        output_dir = self.tr_out_entry.get().strip()
        model_path = self.selected_model_path

        if not os.path.exists(dataset_path):
            alt_ds = os.path.abspath("Finetune/training_data.jsonl")
            if os.path.exists(alt_ds):
                dataset_path = alt_ds
                self.tr_dataset_entry.delete(0, "end")
                self.tr_dataset_entry.insert(0, alt_ds)
            else:
                ctk.messagebox.showerror("Dataset Not Found", f"Training dataset file was not found at:\n{dataset_path}\n\nPlease click Browse to select your training dataset file.")
                return

        batch_size = int(self.tr_batch_var.get())
        lr = float(self.tr_lr_var.get())
        max_steps = int(self.tr_steps_var.get())
        try:
            max_minutes = int(self.tr_time_limit_var.get())
        except ValueError:
            max_minutes = 0

        try:
            grad_accum = int(self.tr_grad_accum_var.get())
        except ValueError:
            grad_accum = 0

        mixed_precision = self.tr_precision_var.get()
        grad_ckpt = (self.tr_grad_ckpt_var.get() == "on")

        self.is_training = True
        self.stop_requested = False

        self.start_tr_btn.configure(state="disabled", text="⏳  Fine-Tuning in Progress...", fg_color="#334155")
        self.stop_tr_btn.configure(state="normal", text="🛑  STOP TRAINING", fg_color="#DC2626")
        self.tr_progress_bar.set(0.0)
        self.tr_progress_lbl.configure(text="Step 0 / 0 | Initializing GPU & Tokenizing Dataset...", text_color="#06B6D4")

        self.tr_log_text.delete("1.0", "end")
        self.tr_log_text.insert("1.0", f"[Launcher] Starting QLoRA fine-tuning run...\n")
        self.tr_log_text.insert("end", f"[Launcher] Base Model: {model_path}\n")
        self.tr_log_text.insert("end", f"[Launcher] Dataset: {dataset_path}\n")
        self.tr_log_text.insert("end", f"[Launcher] Output Directory: {output_dir}\n")
        self.tr_log_text.insert("end", f"[Launcher] Batch Size: {batch_size} | Max Steps: {max_steps}\n")
        self.tr_log_text.insert("end", f"[Launcher] Precision: {mixed_precision} | Grad Accum: {grad_accum if grad_accum > 0 else 'Auto'} | Grad Ckpt: {grad_ckpt}\n\n")

        threading.Thread(
            target=self._training_thread_func,
            args=(model_path, dataset_path, output_dir, batch_size, lr, max_steps, max_minutes, mixed_precision, grad_ckpt, grad_accum),
            daemon=True
        ).start()

    def _training_thread_func(self, model_path, dataset_path, output_dir, batch_size, lr, max_steps, max_minutes, mixed_precision, grad_ckpt, grad_accum):
        old_stdout = sys.stdout
        sys.stdout = OutputRedirector(self.tr_log_text)
        try:
            from Finetune.generic_qlora_trainer import run_qlora_training
            run_qlora_training(
                model_path=model_path,
                config_path="config.yaml",
                dataset_path=dataset_path,
                output_dir=output_dir,
                epochs=1,
                batch_size=batch_size,
                learning_rate=lr,
                max_steps=max_steps,
                sub_sample=45000,
                max_length=384,
                r=16,
                lora_alpha=32,
                dataloader_num_workers=0,
                max_minutes=max_minutes,
                mixed_precision=mixed_precision,
                gradient_checkpointing=grad_ckpt,
                grad_accum=grad_accum,
                stop_checker_fn=lambda: self.stop_requested,
                progress_callback_fn=self._on_training_step_progress
            )
            print("\n[Launcher] QLoRA Fine-Tuning execution completed.")
        except Exception as e:
            print(f"\n[Launcher] Training Exception: {e}")
        finally:
            sys.stdout = old_stdout
            self.after(0, self._on_training_completed)

    def _on_training_step_progress(self, current_step, total_steps, loss, start_time, eff_batch=18):
        def _update():
            TOTAL_DATASET = 4_940_000
            # 1. Run Progress
            if total_steps > 0:
                run_fraction = min(1.0, current_step / float(total_steps))
                run_pct = run_fraction * 100.0
                self.tr_progress_bar.set(run_fraction)
            else:
                run_fraction = 0.0
                run_pct = 0.0

            # 2. Dataset Progress of 4.94M
            records_processed = current_step * eff_batch
            ds_fraction = min(1.0, records_processed / float(TOTAL_DATASET))
            ds_pct = (records_processed / float(TOTAL_DATASET)) * 100.0
            self.tr_dataset_progress_bar.set(ds_fraction)

            # Update Labels
            if hasattr(self, 'run_pct_lbl'):
                self.run_pct_lbl.configure(text=f"{run_pct:.1f}% (Step {current_step:,} / {total_steps:,})")
            if hasattr(self, 'dataset_pct_lbl'):
                self.dataset_pct_lbl.configure(text=f"{ds_pct:.3f}% ({records_processed:,} / 4,940,000 records)")

            elapsed_sec = int(time.time() - start_time)
            m, s = divmod(elapsed_sec, 60)
            h, m = divmod(m, 60)
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

            status_str = f"Step {current_step:,}/{total_steps:,} ({run_pct:.1f}% Run) | {ds_pct:.3f}% of 4.94M Dataset ({records_processed:,} records) | Loss: {loss:.4f} | Elapsed: {elapsed_str}"
            self.tr_progress_lbl.configure(text=status_str, text_color="#10B981")

        self.after(0, _update)

    def _on_training_completed(self):
        self.is_training = False
        self.stop_requested = False
        self.start_tr_btn.configure(state="normal", text="🔥  Start QLoRA Fine-Tuning Process", fg_color="#EF4444")
        self.stop_tr_btn.configure(state="disabled", text="🛑  Stop Training", fg_color="#64748B")
        self.tr_progress_lbl.configure(text="Training finished / stopped. GPU VRAM & cache cleared.", text_color="#94A3B8")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()


if __name__ == "__main__":
    app = ModernRezSLMApp()
    app.mainloop()
