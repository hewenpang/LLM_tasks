import torch
from modelscope import AutoModelForCausalLM,AutoTokenizer
from transformers import  TrainingArguments,Trainer,DataCollatorForSeq2Seq,DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from datasets import load_dataset,load_from_disk
from modelscope import AutoModelForCausalLM, AutoTokenizer


classical_chinese = load_from_disk('/root/data/classical_chinese')
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instraction = tokenizer(
        '\n'.join(['Human: ' + example['instruction'], example['input']]).strip() + '\n\nAssistant: ')
    response = tokenizer(example['output'] + tokenizer.eos_token)

    input_ids = instraction['input_ids'] + response['input_ids']
    attention_mask = instraction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instraction['input_ids']) + response['input_ids']

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


train_data = classical_chinese.map(process_func,remove_columns=classical_chinese.column_names)

lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

data = train_data.train_test_split(0.2,seed=1024)
train_datasets,test_datasets = data['train'],data['test']
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=1000,                # log 打印的频率
    eval_strategy="epoch",
    learning_rate=2e-4,
    logging_dir='./logs',
    output_dir='./outputs',
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_datasets.select(range(2000)),
    eval_dataset=test_datasets.select(range(400)),
    data_collator = collator
)

trainer.train()
model.save_pretrained("./outputs/final_model")import torch
from modelscope import AutoModelForCausalLM,AutoTokenizer
from transformers import  TrainingArguments,Trainer,DataCollatorForSeq2Seq,DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset,load_from_disk
from modelscope import AutoModelForCausalLM, AutoTokenizer


classical_chinese = load_from_disk('/root/data/classical_chinese')
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instraction = tokenizer(
        '\n'.join(['Human: ' + example['instruction'], example['input']]).strip() + '\n\nAssistant: ')
    response = tokenizer(example['output'] + tokenizer.eos_token)

    input_ids = instraction['input_ids'] + response['input_ids']
    attention_mask = instraction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instraction['input_ids']) + response['input_ids']

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


train_data = classical_chinese.map(process_func,remove_columns=classical_chinese.column_names)

lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

data = train_data.train_test_split(0.2,seed=1024)
train_datasets,test_datasets = data['train'],data['test']
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=1000,                # log 打印的频率
    eval_strategy="epoch",
    learning_rate=2e-4,
    logging_dir='./logs',
    output_dir='./outputs',
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_datasets.select(range(2000)),
    eval_dataset=test_datasets.select(range(400)),
    data_collator = collator
)

trainer.train()
model.save_pretrained("./outputs/final_model")



peft_model_path = "./outputs/final_model"
base_model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, peft_model_path)


print('Finetuned_model--------------------------------------------------------------------------------')
prompt = "Human: {}\n{}".format("幫我把文字翻譯成文言文", "我想回家吃饭").strip() + "\n\nAssistant: "
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))


base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
print('Base_model--------------------------------------------------------------------------------------')
model_input = tokenizer('幫我把文字翻譯成文言文\n 我想回家吃饭'.strip()+'\n\nAssistant: ',return_tensors="pt").to(base_model.device)
with torch.no_grad():
    model_output = base_model.generate(**model_input,max_new_tokens=100)
print(tokenizer.decode(model_output[0],skip_special_tokens=True))
