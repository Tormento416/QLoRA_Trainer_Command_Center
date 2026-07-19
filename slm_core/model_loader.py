import os
import yaml
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
bnb_4bit_compute_dtype = "float16"

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
                bnb_4bit_compute_dtype = quant_cfg.get("bnb_4bit_compute_dtype", "float16")
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

# 2. Build BitsAndBytesConfig
bnb_config = None
if load_in_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

print(f"[SLM] Base Model Path = {base_model_path}")
print(f"[SLM] Assistant Model Path = {assistant_model_path}")

# 3. Load Tokenizer & Models
print("[SLM] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("[SLM] Loading base model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map=device_map,
)

assistant_model = None
if assistant_model_path and os.path.exists(assistant_model_path):
    print("[SLM] Loading assistant model for speculative decoding...")
    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
    )
else:
    print("[SLM] Running without assistant model.")

print("[SLM] Load complete.")


# 4. Generation Wrapper
def generate(prompt: str, max_tokens: int = None) -> str:
    if max_tokens is None:
        max_tokens = generation_params["max_new_tokens"]
        
    print(f"[SLM] Generating for prompt (max_tokens={max_tokens})...")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
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
    print("[SLM] Starting model_loader test...")
    test_prompt = "Explain what this local AI assistant can do in one short paragraph."
    try:
        result = generate(test_prompt, max_tokens=128)
        print("[SLM] Output:\n", result)
    except Exception as e:
        print("[SLM] ERROR:", e)