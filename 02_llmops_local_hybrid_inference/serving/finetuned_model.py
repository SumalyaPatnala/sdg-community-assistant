import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class FineTunedModel:
    def __init__(self):
        self.base_model = os.getenv(
            "BASE_MODEL",
            "microsoft/Phi-3-mini-4k-instruct"
        )
        self.adapter_path = os.getenv(
            "ADAPTER_PATH",
            "../03_slm_finetuning_evaluation/outputs/sdg_slm_lora"
        )
        self.model = None
        self.tokenizer = None

    def available(self):
        return os.path.exists(self.adapter_path)

    def load(self):
        if self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

    def generate(self, prompt, max_tokens=512):
        self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)