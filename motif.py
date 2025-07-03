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

#### Defining the system prompt for MOTIF
SYSTEM_PROMPT = """
You are a helpful assistant. When the user asks a question, you solve it in 3 rounds.
In each round, you first think about the reasoning process of answering and then provide the user with a detailed progress about it.
The reasoning process and the progress are enclosed within <reasoning> </reasoning> and <answer> </answer> tags respectively.
Therefore, you follow the strict format:
<reasoning>
reasoning process here
</reasoning>
<answer>
detailed progress here
</answer>

The User provides this detailed progress as additional context in the next round.
You then respond again with further thinking and further progress.
When the User says that current round is the final (third) round, you provide an answer inside the answer tags.
You also enclose a final answer in third round in the box: \\boxed{}. Only this boxed final answer is used for evaluation.
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
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]: #dummy reward, defined as a placeholder only
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [900.0 for r, a in zip(extracted_responses, answer)]

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
    run_name = "motif_gsm8k",
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
    max_steps = 1000,
    save_steps = 100,
    max_grad_norm = 0.1,
    # report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
    beta = 0.0,
)

#### Defining the RL trainer
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [correctness_reward_func, xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func], # correctness is dummy reward
    args = training_args,
    train_dataset = train_data,
)



#### Model iterations
# max_iter = 3 # maximum number of iterations

def model_iteration(model_def, prompt, init_response, t_num, temp = 0.8, top_p_value = 0.95, max_tokens_num = 2048, max_iter = 3, lora_def = None) -> list[str]:
    final_responses = [] # actually final_responses
    for i in range(t_num):
        ans  = init_response
        iter = 1
        while iter < max_iter:
            ans = extract_xml_answer(ans)
            augmented_prompt = prompt + f" Progress in round {iter}: \"" + ans + "\""
            iter = iter + 1
            if iter == max_iter:
                augmented_prompt = augmented_prompt + f" Current round is the final (third) round. Provide a final answer."
            messages = [{"role" : "system", "content" : SYSTEM_PROMPT}, {"role" : "user", "content" : augmented_prompt},]
            text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
            sampling_params = SamplingParams(temperature = temp, top_p = top_p_value, max_tokens = max_tokens_num)
            with torch.no_grad():
                ans = model_def.fast_generate(text, sampling_params = sampling_params, lora_request = lora_def,)[0].outputs[0].text
        # ans = extract_xml_answer(ans)  #considering whole response
        final_responses = final_responses + [ans]
    return final_responses


#### MOTIF reward function for multi round inference
traj_num = 4 # number of trajectories generated

def total_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}")
    #iterative model inference to calculate future accuracy rewards
    final_responses_list = []
    final_answers_list = []
    for r in responses:
        # print(r)
        final = model_iteration(trainer.model, q, r, traj_num)
        final_responses_list = final_responses_list + final
        for entry in final:
            final_answers_list = final_answers_list + [extract_xml_answer(entry)]

    answer_mod = [item for item in answer for i in range(traj_num)] # answer = answer * traj_num ????

    total_rewards = [2.0 if f'\\boxed{{{a}}}' in f else 1.0 if f'\\boxed{{{a}}}' in r else 0.0 for f, r, a in zip(final_answers_list, final_responses_list, answer_mod)]
    compressed_rewards = [sum(total_rewards[i*traj_num:(i+1)*traj_num])/traj_num for i in range(len(answer))]
    return compressed_rewards

#### Redefining the trainer reward functions to incorporate correctness reward from the model iterations
trainer.reward_funcs = [total_reward_func, xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func]
print(trainer.reward_funcs)



######## Training and Evaluation

#### Training the model
training_rounds = 1
for i in range(training_rounds):
    trainer.train()
    model.save_lora(f"motif_saved_lora_round_{i+1}")


#### Benchmark evaluations
## MATH500
test_data = load_dataset('HuggingFaceH4/MATH-500')['test']
test_num = 500
start_index = 0
training_rounds = 1

response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT},
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
            lora_request = model.load_lora(f"motif_saved_lora_round_{training_rounds}"),
        )[0].outputs[0].text
        [final_response] = model_iteration(model, question, ans, t_num = 1, temp = 0, top_p_value = 1.0, max_tokens_num = 2048, lora_def=model.load_lora(f"motif_saved_lora_round_{training_rounds}"))
    print(count)
    count+=1
    response_list.append(final_response)
# accuracy calculation
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
print(f"MOTIF MATH500 accuracy: {acc_pecent*100}%")

## AIME2024
test_data = load_dataset("HuggingFaceH4/aime_2024")['train']
test_num = 30
start_index = 0

response_list = []
count=1
for question in test_data['problem'][start_index:start_index+test_num]:
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT},
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
            lora_request = model.load_lora(f"motif_saved_lora_round_{training_rounds}"),
        )[0].outputs[0].text
        [final_response] = model_iteration(model, question, ans, t_num = 1, temp = 0, top_p_value = 1.0, max_tokens_num = 2048, lora_def=model.load_lora(f"motif_saved_lora_round_{training_rounds}"))
    print(count)
    count+=1
    response_list.append(final_response)

# acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if f'\\boxed{{{y}}}' in x) / len(response_list)
acc_pecent = sum(1 for x,y in zip(response_list, test_data['answer'][start_index:start_index+test_num]) if y in x) / len(response_list)
print(f"MOTIF AIME2024 accuracy: {acc_pecent*100}%")