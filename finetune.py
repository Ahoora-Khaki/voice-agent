"""
Fine-tune Gemma 2 2B with DoRA on labeled training data
Optimized for Apple Silicon (MPS) with 16GB RAM
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "google/gemma-2-2b-it"
OUTPUT_DIR = "/Users/ahoora/voice-agent/adapters/savage-helper"
DATA_PATH = "/Users/ahoora/voice-agent/data/train.jsonl"

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
WARMUP_STEPS = 5

# DoRA configuration (note: use_dora=True)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_DORA = True  # 🔥 Enable DoRA

# ============================================================
# LOAD DATA
# ============================================================

def load_training_data(data_path):
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append(data)
    print(f"✅ Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)

# ============================================================
# FORMAT FOR GEMMA
# ============================================================

def format_example(example, tokenizer):
    user_content = f"[{example['type']}] {example['instruction']}\n\nUser: {example['input']}"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example['output']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": text}

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("🚀 SAVAGE HELPER FINE-TUNING (DoRA)")
    print("="*60)
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print(f"⚠️ MPS not available, using CPU")
    
    # Load tokenizer
    print("\n📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("📦 Loading Gemma 2B (2-3 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    
    model.gradient_checkpointing_enable()
    
    # Configure DoRA
    print("🔧 Configuring DoRA adapters...")
    dora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=USE_DORA,  # 🔥 DoRA enabled
    )
    
    model = get_peft_model(model, dora_config)
    
    print("\n📊 Trainable parameters:")
    model.print_trainable_parameters()
    
    # Load and format data
    print("\n📊 Loading training data...")
    dataset = load_training_data(DATA_PATH)
    
    print("🔄 Formatting examples...")
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    def tokenize_function(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = dataset.map(tokenize_function, remove_columns=["text"])
    print(f"✅ Dataset ready: {len(dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        optim="adamw_torch",
        report_to="none",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*60)
    print("🏋️ STARTING DoRA TRAINING")
    print("="*60)
    print(f"   Model: {MODEL_NAME}")
    print(f"   Method: DoRA (Weight-Decomposed LoRA)")
    print(f"   Examples: {len(dataset)}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Rank: {LORA_R}")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save
    print("\n💾 Saving DoRA adapters...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("🎉 DoRA TRAINING COMPLETE!")
    print(f"   Adapters saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
