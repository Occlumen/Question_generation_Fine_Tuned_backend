import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel, get_peft_model

token = "enter_your_token"
model_name = "google-bert/bert-large-uncased"

# Load the dataset and tokenizer
dataset = load_dataset("squad")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

# Load the base model
base_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    task_type="QUESTION_ANSWERING", # Task type
    inference_mode=False,           # Enable training mode
    r=2,                            # Rank of LoRA
    lora_alpha=4,                  # LoRA scaling factor
    lora_dropout=0.2,               # Dropout rate for LoRA layers
)

# Wrap the base model with PEFT
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters for debugging

def preprocess(batch):
    inputs = tokenizer(
        batch["question"],
        batch["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
    )

    # Calculate start and end positions for each example in the batch
    start_positions = []
    end_positions = []

    for i in range(len(batch["answers"])):
        if len(batch["answers"][i]["answer_start"]) > 0:
            start_pos = batch["answers"][i]["answer_start"][0]
            end_pos = start_pos + len(batch["answers"][i]["text"][0])
        else:
            # Default positions for unanswerable questions
            start_pos = 0
            end_pos = 0

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    # Update inputs with start and end positions
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

# Preprocess the reduced datasets
train_data = dataset["train"].map(preprocess, batched=True)
validation_data = dataset["validation"].map(preprocess, batched=True)

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory for saving results
    evaluation_strategy="epoch",    # Evaluate after each epoch
    learning_rate=2e-4,             # Learning rate (can be higher with PEFT)
    per_device_train_batch_size=1, # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    num_train_epochs=5,             # Number of epochs
    weight_decay=0.02,              # Weight decay
    save_total_limit=2,             # Save only the latest checkpoints
    fp16=True,                      # Use mixed precision training (optional)
    report_to="none",  # Disable W&B logging
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                     # LoRA-enabled model
    args=training_args,              # Training arguments
    train_dataset=train_data,        # Training dataset
    eval_dataset=validation_data     # Validation dataset
)

trainer.train()

model.save_pretrained("./fine_tuned_bert_large")
tokenizer.save_pretrained("./fine_tuned_bert_large")

