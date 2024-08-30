import torch
from transformers import GenerationConfig

from qwen.modeling_qwen import QWenLMHeadModel
from qwen.tokenization_qwen import QWenTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelpath = "/home/caslx/scalinglaw/MINI_LLM/model_save/sft/checkpoint-42000/"
# modelpath = "/home/caslx/scalinglaw/MINI_LLM/ktosavemodel/checkpoint-36092/"
# modelpath = "/home/caslx/scalinglaw/MINI_LLM/dposavemodel/checkpoint-9400/"
# modelpath = "/home/caslx/scalinglaw/MINI_LLM/model_save/pre/checkpoint-42400/"
model = QWenLMHeadModel.from_pretrained(modelpath).to(device)
tokenizer = QWenTokenizer.from_pretrained(modelpath)
"""
长城有多长呢 
世界上最大的动物是什么
苹果公司是什么 




"""
gen_config = GenerationConfig(
    temperature=0.3,
    top_k=20,
    top_p=0.5,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=300,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print("程序启动")


while True:
    print("我：", end="")
    text = input()

    # print(text)
    text = "你是一个助手 用户: {text} 回答: ".format(text=text)
    tokend = tokenizer(text, add_special=False)
    input_ids, attention_mask = torch.LongTensor([tokend.input_ids]).to(
        device
    ), torch.LongTensor([tokend.attention_mask]).to(device)
    outputs = model.generate(
        inputs=input_ids, attention_mask=attention_mask, generation_config=gen_config
    )
    outs = tokenizer.decode(outputs[0].cpu().numpy())
    print("AI:" + outs.split("<|im_end|>")[0].split("回答: ")[1])
