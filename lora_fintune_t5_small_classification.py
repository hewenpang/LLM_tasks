from datasets import load_from_disk
from modelscope import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from transformers.pipelines.pt_utils import KeyDataset
# 加载数据集
data = load_from_disk('/root/data/rotten_tomatos')
# #
# # # 0/1 映射到文本标签
# label_map = {0: "negative", 1: "positive"}
#
# # 加载模型和tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained('AI-ModelScope/t5-small')
# tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/t5-small')

# 处理函数，转文本标签token id作为labels
# def process(example):
#     prompts = [
#         "Is the following sentence positive or negative? " + text
#         for text in example["text"]
#     ]
#
#     # 输入编码
#     inputs = tokenizer(
#         prompts,
#         padding="max_length",
#         truncation=True,
#         max_length=128,
#     )
#
#     # 标签文本
#     label_texts = [label_map[label] for label in example["label"]]  # 注意用原始标签字段
#
#     # 标签编码
#     labels = tokenizer(
#         label_texts,
#         padding="max_length",
#         truncation=True,
#         max_length=8,
#     )["input_ids"]
#
#
#     labels = [
#         [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
#         for label_seq in labels
#     ]
#
#     return {
#         "input_ids": inputs["input_ids"],
#         "attention_mask": inputs["attention_mask"],
#         "label": labels,
#     }
#
# # 应用数据处理
# tokenized_data = data.map(process, batched=True)
#
#
# # print(tokenized_data['train'][0]['labels'])
#
# # LoRA 配置
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["q", "v"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.SEQ_2_SEQ_LM
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
#
# # DataCollator
# collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
#
# # 训练参数
# training_args = TrainingArguments(
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     logging_steps=1000,
#     eval_strategy="epoch",
#     learning_rate=2e-4,
#     logging_dir='./logs',
#     output_dir='./outputs'
# )
#
# # Trainer
# trainer = Trainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     train_dataset=tokenized_data['train'],
#     eval_dataset=tokenized_data['validation'],
#     data_collator=collator
# )
#
# trainer.train()
# model.save_pretrained("./outputs/text_classification")



peft_model_path = "./outputs/text_classification"
base_model_name = "AI-ModelScope/t5-small"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, peft_model_path)

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

from transformers import pipeline
pipe = pipeline(
    "text2text-generation",
    model= model,
    tokenizer = tokenizer
)
prompt = 'Is the following sentence positive or negative? '
data = data.map(lambda x:{'t5':prompt+x['text']})
from tqdm import tqdm

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)

evaluate_performance(data["test"]["label"], y_pred)


base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
pipe2 = pipeline(
"text2text-generation",
    model = base_model,
    tokenizer = tokenizer
)
y_pred2 = []
for output in tqdm(pipe2(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred2.append(0 if text == "negative" else 1)
evaluate_performance(data["test"]["label"], y_pred2)