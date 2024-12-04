import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

token = os.environ.get('hf_api_token')

os.environ["WANDB_DISABLED"] = "true"

dataset_train = load_dataset("json", data_files={"train": "/kaggle/input/squad-dataset/train-v2.0.json"})
dataset_validation = load_dataset("json", data_files={"validation": "/kaggle/input/squad-dataset/dev-v2.0.json"})

# Load tokenizer and model
model_name = "/kaggle/input/mistral-instruct-three"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,token=token)
base_model = AutoModel.from_pretrained(model_name,token=token)

os.environ["WANDB_DISABLED"] = "true"

# Define the LoRALayer class (as per your requirements)
class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=1, alpha=2):
        super(LoRALayer, self).__init__()
        # Replace with LoRA-specific initialization logic
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        # Replace with LoRA-specific forward logic
        return self.base_layer(x)

# Function to apply LoRA to the model
def apply_lora_to_model(model, rank=1, alpha=2):
    # Collect the names of linear layers
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers_to_replace.append(name)

    # Replace the layers in a second loop
    for name in layers_to_replace:
        # Traverse the module hierarchy to replace the layer
        parent_module = model
        submodule_names = name.split('.')
        for sub_name in submodule_names[:-1]:
            parent_module = getattr(parent_module, sub_name)

        # Replace the target layer with LoRALayer
        original_layer = getattr(parent_module, submodule_names[-1])
        setattr(parent_module, submodule_names[-1], LoRALayer(original_layer, rank, alpha))

    return model

# Apply LoRA to the model
model = apply_lora_to_model(base_model)

def preprocess_function(examples):
    # Initialize lists to store processed data
    questions = []
    contexts = []
    start_positions = []
    end_positions = []

    # Set padding token if it does not exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Iterate over each example in the dataset
    for example in examples["data"][0]:
        if "paragraphs" in example and isinstance(example["paragraphs"], list):
            for paragraph in example["paragraphs"]:
                context = paragraph.get("context", "")  # Extract the context safely

                if "qas" in paragraph and isinstance(paragraph["qas"], list):
                    for qa in paragraph["qas"]:
                        question = qa.get("question", "")  # Extract the question safely

                        # Handle answers
                        if not qa.get("is_impossible", True):
                            if "answers" in qa and isinstance(qa["answers"], list) and len(qa["answers"]) > 0:
                                answer = qa["answers"][0]
                                start_position = answer.get("answer_start", 0)
                                end_position = start_position + len(answer.get("text", ""))
                            else:
                                start_position = 0
                                end_position = 0
                        else:
                            start_position = 0
                            end_position = 0

                        # Append extracted data to lists
                        questions.append(question)
                        contexts.append(context)
                        start_positions.append(start_position)
                        end_positions.append(end_position)

    # Debugging output
    print(f"Number of questions: {len(questions)}")
    print(f"Number of contexts: {len(contexts)}")

    # Check if lists are empty
    if not questions or not contexts:
        print("Questions or contexts are empty.")
        return {"input_ids": [], "attention_mask": [], "start_positions": [], "end_positions": []}

    # Ensure both lists have the same length
    if len(questions) != len(contexts):
        print("Mismatch in lengths of questions and contexts.")
        return {"input_ids": [], "attention_mask": [], "start_positions": [], "end_positions": []}

    # Tokenize questions and contexts
    inputs = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=1536,
        return_tensors='np'  # Use numpy arrays for better compatibility
    )

    # Convert to lists
    input_ids = inputs['input_ids'].tolist()
    attention_mask = inputs['attention_mask'].tolist()

    # Check lengths
    assert len(input_ids) == len(attention_mask) == len(start_positions) == len(end_positions), "Mismatch in lengths after tokenization."

    # Return the tokenized and processed data
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_positions": start_positions,
        "end_positions": end_positions
    }

# Limit the number of entries to 5000
max_entries = 5000
limited_train_dataset = dataset_train["train"].select(range(min(max_entries, len(dataset_train["train"]))))
limited_validation_dataset = dataset_validation["validation"].select(range(min(max_entries, len(dataset_validation["validation"]))))

# Preprocess the train and validation datasets
tokenized_train_dataset = limited_train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = limited_validation_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # To suppress W&B or other integrations
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()


model.save_pretrained("./fine_tuned_Mistral-7B-Instruct-v0.2_QA_Dataset")
tokenizer.save_pretrained("./fine_tuned_Mistral-7B-Instruct-v0.2_QA_Dataset")
