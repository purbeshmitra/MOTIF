#### Some of the functions in the code are directly adapted from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb and https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import torch
import re

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

#### Defining the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,#True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    # gpu_memory_utilization = 0.5, # Reduce if out of memory
)

#### Defining the LoRA adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

#### Defining the system prompt for GRPO
SYSTEM_PROMPT = """You are a helpful assistant.
When the user asks a question, you first think about the reasoning process in mind and then provide the user with an answer.
The reasoning process and the answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively.
In your answer, you also enclose your final answer in the box: \\boxed{}.
Therefore, you respond in the following strict format:
<reasoning>
reasoning process here
</reasoning>
<answer>
answer here
</answer>
"""


#### Load and prep the dataset
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

train_data = get_gsm8k_questions()


#### Defining functions for RL training
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

## Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}")
    return [2.0 if f'\\boxed{{{a}}}' in r else 0.0 for r, a in zip(responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


#### Defining the RL training arguments
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    run_name = "grpo_gsm8k",
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 2000,
    save_steps = 250,
    max_grad_norm = 0.1,
    # report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    beta= 0.0,
)

#### Defining the RL trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = train_data,
)


######## Training and Evaluation

#### Training the model
training_rounds = 1
for i in range(training_rounds):
    trainer.train()
    model.save_lora(f"grpo_saved_lora_round_{i+1}")

SYSTEM_PROMPT_base = "You are a helpful assistant. In your response, you enclose your final answer in the box: \\boxed{}."
SYSTEM_PROMPT_grpo = SYSTEM_PROMPT


#### Benchmark evaluations
## MATH500
test_data = load_dataset('HuggingFaceH4/MATH-500')['test']
test_num = 500
start_index = 0

response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT_base},
        {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    sampling_params = SamplingParams(
        temperature = 0,
        top_p = 1.0,
        max_tokens = 1024,
    )
    with torch.no_grad():
        ans = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = None,
        )[0].outputs[0].text
    print(count)
    count+=1
    print(ans)
    response_list.append(ans)
# accuracy calculation for base model
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
print(f"Base model MATH500 accuracy: {acc_pecent*100}%")

response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT_grpo},
        {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    sampling_params = SamplingParams(
        temperature = 0,
        top_p = 1.0,
        max_tokens = 1024,
    )
    with torch.no_grad():
        ans = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = model.load_lora(f"grpo_saved_lora_round_{training_rounds}"),
        )[0].outputs[0].text
    print(count)
    count+=1
    response_list.append(ans)
# accuracy calculation for GRPO trained model
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
print(f"GRPO trained model MATH500 accuracy: {acc_pecent*100}%")


## AIME2024
test_data = load_dataset("HuggingFaceH4/aime_2024")['train']
test_num = 30
start_index = 0

response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT_base},
        {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    sampling_params = SamplingParams(
        temperature = 0,
        top_p = 1.0,
        max_tokens = 1024,
    )
    with torch.no_grad():
        ans = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = None,
        )[0].outputs[0].text
    print(count)
    count+=1
    response_list.append(ans)
# accuracy calculation for base model
# acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if y in x) / len(response_list)
print(f"Accuracy: {acc_pecent*100}%")


response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        # {"role" : "system", "content" : SYSTEM_PROMPT_grpo},
        {"role" : "user", "content" : question},
    ], tokenize = False, add_generation_prompt = True)
    sampling_params = SamplingParams(
        temperature = 0,
        top_p = 1.0,
        max_tokens = 1024,
    )
    with torch.no_grad():
        ans = model.fast_generate(
            text,
            sampling_params = sampling_params,
            lora_request = model.load_lora(f"grpo_saved_lora_round_{training_rounds}"),
        )[0].outputs[0].text
    print(count)
    count+=1
    response_list.append(ans)
# accuracy calculation for GRPO trained model
# acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if y in x) / len(response_list)
print(f"Accuracy: {acc_pecent*100}%")