from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import json

# ----------------------------------------------------------------------------------------
# hugging face model name
model_name = 'FacebookAI/xlm-roberta-base'

# languages
languages = 'en-it'

# batch size
batch_size = 32

# epochs
epochs = 6

# learning rate
learning_rate = 5e-5

# use cpu (set false for nvidia gpu)
use_cpu = False
# ----------------------------------------------------------------------------------------

# output folder name
experiment_name = f'{model_name}-fine-tuned-{languages}-{epochs}epochs-{batch_size}batch'

# load dataset
dataset = load_dataset(
    'parquet',
    data_files={
        'train': f'data/multi-label-classification/train-{languages}.parquet',
        'validation': f'data/multi-label-classification/validation-{languages}.parquet',
        'test': f'data/multi-label-classification/test-{languages}.parquet'
    }
)

# load labels mapping
with open('data/multi-label-classification/labels-mapping-en-it.json', 'r') as file:
    label2id = json.load(file)
id2label = {id: label for label, id in label2id.items()}

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenize function
def tokenize_text(examples):   
    return tokenizer(examples['text'], truncation=True)

# tokenize dataset  
tokenized_dataset = dataset.map(tokenize_text, batched=True)

# data collator for dynamically padding batch instead of padding whole dataset to max length
# this can speed up considerably the training procedure if batches samples have a short text length
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    problem_type='multi_label_classification',
    id2label=id2label,
    label2id=label2id
)

# training arguments
training_args = TrainingArguments(
    output_dir=f'checkpoints/multi-label-classification/{experiment_name}-checkpoints',
    overwrite_output_dir=True,
    logging_strategy='epoch',
    eval_strategy='epoch',    
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_delay=0,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    use_cpu=use_cpu,
    report_to='none'
)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,    
)

# start training
trainer.train()

# save model
trainer.save_model(f'models/multi-label-classification/{experiment_name}-model')
