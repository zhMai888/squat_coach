import json

import torch
from modelscope import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load():
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

    return tokenizer, model


class Prompt():
    def __init__(self):
        self.prompt = ""

        # prompt开头
        self.prompt_header = (
            "你是一位深蹲动作评估专家。下面输入的是某段视频中多次深蹲的动作数据。"
            "输入为一个三维数组，格式为(n,5,8,4)。\n"
            "n表示深蹲的次数，5表示每次深蹲的关键帧数（站立开始帧，向下蹲随机帧，蹲到最下帧，向上站随机帧，站立结束帧）"
            "每帧有8个维度（躯干角度，髋角度，膝盖到脚尖的距离，脚跟稳定性，深蹲深度，肩部对称性，膝盖对称性，膝盖宽度），"
            "每个维度包含4个数值，分别表示：分数、置信度、是否标准（1为标准，0为不标准）、偏置量。\n"
            "请基于这些数据，逐次评价每次深蹲的整体表现。采用逐步推理的方法，按以下思路进行分析：\n"
            "1. 分析每帧数据，判断动作的稳定性和一致性。\n"
            "2. 识别每帧中动作的主要问题（如低置信度、不标准等）。\n"
            "3. 针对每个维度（如躯干角度、髋角度、膝盖距离等），逐一分析并解释可能原因。\n"
            "4. 提出改进建议，针对每个问题提供具体建议。\n"
            "最后，请给出对所有深蹲动作的总体评价，指出共性问题和改进重点。\n"
            "以下是深蹲数据：\n\n"
        )

        self.prompt += self.prompt_header

    # 加载prompt
    def load_prompt(self,file_path="fewshot.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data

    # prompt函数
    def generate_prompt(self, data, first_time=False):
        prompt = ""
        if first_time:
            prompt = self.prompt
        else:
            prompt = "以下是深蹲数据：\n"

        prompt += str(data[0]["data"])

        prompt += "\n\n"+"示例评价："+str(data[0]["evaluations"]).strip('[]').replace("'", "")+"\n"+"整体评价："+str(data[0]["overall_evaluation"]).strip('[]').replace("'", "")

        return prompt

    # 预测prompt
    def predict_prompt(self, data):
        prompt = self.prompt_header          # 防止遗忘规则
        prompt += str(data)+"\n"
        prompt += "一共有" + str(len(data)) + "次深蹲"

        prompt += "\n\n"+"请参考之前的示例评价格式给这几次深蹲进行评价。记住，是每次深蹲的表现以及整体表现。"

        return prompt


if __name__ == '__main__':
    a = Prompt()
    data = a.load_prompt()
    # tokenizer, model = load()
    history = None
    for i in range(len(data.keys())):
        if i == 0:
            p = a.generate_prompt(data["fewshot"+str(i+1)], first_time=True)
        else:
            p = a.generate_prompt(data["fewshot"+str(i+1)])

        print(p)
        # response, history = model.chat(tokenizer, p, history=history)
        # print(len(tokenizer.encode(str(history))))
        # print(response)
