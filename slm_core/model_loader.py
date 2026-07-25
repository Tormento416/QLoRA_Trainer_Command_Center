import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Default paths and configurations (used if config.yaml is missing or has errors)
CONFIG_PATH = "config.yaml"
base_model_path = r"D:\models"
assistant_model_path = r"D:\models\gemma-4-transformers-gemma-4-e4b-it-assistant-v1"
device_map = "auto"

# Default quantization settings
load_in_4bit = True
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.float16

# Default generation parameters
generation_params = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "do_sample": True,
}

# 1. Parse configuration from config.yaml
if os.path.exists(CONFIG_PATH):
    print(f"[SLM] Loading configuration from {CONFIG_PATH}...")
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = yaml.safe_load(f)
            
            # Read model paths
            model_cfg = config_data.get("model", {})
            adapter_path = model_cfg.get("adapter_path", None)
            if "base_path" in model_cfg:
                base_model_path = model_cfg["base_path"]
            if "assistant_path" in model_cfg:
                assistant_model_path = model_cfg["assistant_path"]
            device_map = model_cfg.get("device_map", "auto")
            
            # Read quantization parameters
            quant_cfg = config_data.get("quantization", {})
            if quant_cfg.get("enabled", True):
                load_in_4bit = quant_cfg.get("load_in_4bit", True)
                bnb_4bit_use_double_quant = quant_cfg.get("bnb_4bit_use_double_quant", True)
                bnb_4bit_quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
                dtype_val = quant_cfg.get("bnb_4bit_compute_dtype", "float16")
                bnb_4bit_compute_dtype = getattr(torch, dtype_val, torch.float16) if isinstance(dtype_val, str) else dtype_val
            else:
                load_in_4bit = False
                
            # Read generation parameters
            gen_cfg = config_data.get("generation", {})
            if gen_cfg:
                generation_params["max_new_tokens"] = gen_cfg.get("max_new_tokens", 256)
                generation_params["temperature"] = gen_cfg.get("temperature", 0.2)
                generation_params["top_p"] = gen_cfg.get("top_p", 0.9)
                generation_params["top_k"] = gen_cfg.get("top_k", 40)
                generation_params["do_sample"] = gen_cfg.get("temperature", 0.2) > 0
    except Exception as e:
        print(f"[SLM] Error parsing config.yaml: {e}. Falling back to defaults.")

# 2. Check Device & Build Quantization Config
is_cuda_available = torch.cuda.is_available()
if not is_cuda_available:
    print("[SLM] Warning: CUDA is not available. Disabling 4-bit bitsandbytes quantization for CPU execution.")
    load_in_4bit = False
    device_map = "cpu"

bnb_config = None
model_kwargs = {"device_map": device_map}

if load_in_4bit and is_cuda_available:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    model_kwargs["quantization_config"] = bnb_config
else:
    model_kwargs["dtype"] = torch.float32

print(f"[SLM] Base Model Path = {base_model_path}")
print(f"[SLM] Assistant Model Path = {assistant_model_path}")

# Global references for lazy loading
tokenizer = None
model = None
assistant_model = None
is_model_loaded = False

def ensure_model_loaded():
    global tokenizer, model, assistant_model, is_model_loaded
    if is_model_loaded and model is not None:
        return

    print(f"[SLM] Base Model Path = {base_model_path}")
    print(f"[SLM] Assistant Model Path = {assistant_model_path}")

    # Load Tokenizer
    print("[SLM] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load Base Model
    print("[SLM] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )

    if 'adapter_path' in locals() and adapter_path and os.path.exists(adapter_path):
        print(f"[SLM] Loading fine-tuned LoRA adapter from {adapter_path}...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print("[SLM] LoRA adapter successfully attached.")
        except Exception as e:
            print(f"[SLM] Failed to load LoRA adapter: {e}")

    if assistant_model_path and os.path.exists(assistant_model_path):
        print("[SLM] Loading assistant model for speculative decoding...")
        try:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                assistant_model_path,
                **model_kwargs,
            )
        except Exception as e:
            print(f"[SLM] Note: Skipping assistant model load: {e}")
            assistant_model = None

    is_model_loaded = True
    print("[SLM] Model load complete.")

from slm_core.memory import ShortTermMemory, LongTermMemory

# 4. Generation Wrapper
def generate(prompt_or_messages, max_tokens: int = None) -> str:
    ensure_model_loaded()

    if max_tokens is None:
        max_tokens = generation_params["max_new_tokens"]
        
    print(f"[SLM] Generating response (max_tokens={max_tokens})...")
    
    # Format messages array or single string prompt
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Configure generation parameters from config.yaml
    gen_args = {
        "max_new_tokens": max_tokens,
        "temperature": generation_params["temperature"],
        "top_p": generation_params["top_p"],
        "top_k": generation_params["top_k"],
        "do_sample": generation_params["do_sample"],
    }
    
    # Enable speculative decoding if assistant model is available
    if assistant_model:
        gen_args["assistant_model"] = assistant_model
        
    outputs = model.generate(**inputs, **gen_args)
    
    # Decode only the generated response (excluding the prompt text)
    input_len = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Local SLM Assistant Ready (with Memory Bank)")
    print("  Commands:")
    print("    /remember key=value  : Save a fact to long-term memory")
    print("    /memories            : List all saved facts")
    print("    /clear               : Clear short-term conversation context")
    print("    exit / quit          : End session")
    print("=" * 50)

    # Initialize memory
    mem_cfg = config_data.get("memory", {}) if os.path.exists(CONFIG_PATH) else {}
    max_turns = mem_cfg.get("max_turns", 10)
    storage_path = mem_cfg.get("storage_path", "data/memory_bank.json")

    short_mem = ShortTermMemory(max_turns=max_turns)
    long_mem = LongTermMemory(storage_path=storage_path)

    base_system_prompt = "You are a helpful local AI assistant."

    while True:
        try:
            user_input = input("\nHow can I help you today? > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("[SLM] Goodbye!")
                break
            if user_input.lower() == "/clear":
                short_mem.clear()
                print("[SLM] Short-term conversation history cleared.")
                continue
            if user_input.lower() == "/memories":
                print("\n[Memory Bank Stored Facts]:")
                if not long_mem.memories:
                    print("  (No facts saved yet)")
                else:
                    for k, v in long_mem.memories.items():
                        print(f"  - {k}: {v}")
                continue
            if user_input.startswith("/remember "):
                fact_str = user_input[10:].strip()
                if "=" in fact_str:
                    k, v = fact_str.split("=", 1)
                else:
                    k, v = fact_str, fact_str
                long_mem.add_fact(k, v)
                print(f"[SLM] Saved to memory bank: '{k.strip()}' = '{v.strip()}'")
                continue

            # Add user message to short-term memory
            short_mem.add_user_message(user_input)

            # Build system prompt with relevant long-term memory bank facts
            system_prompt = long_mem.build_system_prompt(base_system_prompt, user_input)
            
            # Prepare messages with system prompt + short-term history
            messages = short_mem.get_messages(system_prompt=system_prompt)

            # Generate response
            response = generate(messages)

            # Record assistant turn in short-term memory
            short_mem.add_assistant_message(response)

            print(f"\n[Assistant]: {response}")
        except KeyboardInterrupt:
            print("\n[SLM] Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\n[SLM] ERROR: {e}")