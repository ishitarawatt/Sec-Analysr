"""QLoRA fine-tune for structured financial extraction. Needs a CUDA GPU (e.g. 1x 24GB).

  pip install -r finetune/requirements-gpu.txt
  python -m finetune.build_dataset
  python finetune/qlora_train.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2

After training: merge adapters, evaluate with finetune/eval_extraction.py against the base model
and a hosted API, then serve with vLLM and point VLLM_BASE_URL at it."""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ds = load_dataset("json", data_files={"train": "finetune/data/train.jsonl",
                                          "test": "finetune/data/test.jsonl"})
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                    per_device_train_batch_size=4, gradient_accumulation_steps=4,
                    logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
                    bf16=True, max_length=2048, report_to="none")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"], eval_dataset=ds["test"],
                         peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
