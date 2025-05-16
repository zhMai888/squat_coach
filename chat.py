import torch
from modelscope import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 使用BitsAndBytesConfig加载8位模型
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,  # 关闭4位
    load_in_8bit=True,   # 启用8位
    llm_int8_enable_fp32_cpu_offload=True  # 启用FP32 CPU Offload
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained("../Qwen-1_8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "../Qwen-1_8B-Chat",
    trust_remote_code=True,
    device_map = device,
    quantization_config=quantization_config
).eval()

model.generation_config = GenerationConfig.from_pretrained("../Qwen-1_8B-Chat", trust_remote_code=True)

response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "狮子是什么？", history=history)
print(response)
