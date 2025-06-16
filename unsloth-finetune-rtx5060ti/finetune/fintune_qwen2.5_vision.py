from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset
import os
import shutil

save_merged=False
model_id = "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit"
dataset_id = "5CD-AI/Viet-Doc-VQA-verIII"
output = "outputs/qwen2.5-vision-vietdocvqa-3b-4bit"
adapter= ""
merged_dir= output + "/merged-model"
dataset_start = 0
dataset_end = 1000



# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

if adapter=="":
    print("No adapter specified, using pre-trained model")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        max_seq_length=4096, # Set to 4096 for long context
    )
        
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
else:
    print(f"Using adapter: {adapter}")
    model, tokenizer = FastVisionModel.from_pretrained(
        #model_id,
        adapter, # Use a pre-trained adapter
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        max_seq_length=4096
        
    )
# Load dataset
print(f"Load dataset from {dataset_id} with start={dataset_start} and end={dataset_end}")
dataset = load_dataset(
    dataset_id,
    split = "train",
).select(range(dataset_start, dataset_end))

def convert_to_conversation(example):
    messages = []
    messages.append({
        "role": "user", 
        "content": [
            {"type": "text", "text": "Hãy trích xuất nội dung của bức ảnh sau dưới dạng markdown."},
            {"type": "image", "image": example["image"]},
            ]
        })
    messages.append({
        "role": "assistant", 
        "content": [
            {"type": "text", "text": example["description"]}
        ]})
    for conversation in example["conversations"]:
        messages.append({
            "role": conversation["role"], 
            "content": [
                {"type": "text", "text": conversation["content"]},
            ]})
    return { "messages": messages }

print(f"Converting dataset to conversation format with {len(dataset)} samples")
formated_dataset = [convert_to_conversation(sample) for sample in dataset]

print(f"Example conversation: {formated_dataset[456]}")

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

print("Creating trainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = formated_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output,
        report_to = "none",     # For Weights and Biases
        save_strategy = "steps", 
        save_steps = 20,      
        save_total_limit = 1,  # Keep only the last 2 checkpoints
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 4096,
        max_length= 4096,
    ),
)

resume_from_checkpoint = False # Set to "latest" to resume training
for item in os.listdir(output):
    if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output, item)):
        print(f"Found checkpoint: {item}")
        resume_from_checkpoint = True
        break

print("Starting training...")
trainer_stats = trainer.train(resume_from_checkpoint = resume_from_checkpoint) # Set to "latest" to resume training

print("Training completed!")
# save adapter
print("Saving model and tokenizer...")
trainer.model.save_pretrained(output)  # Local saving
trainer.tokenizer.save_pretrained(output)
# Merge LoRA adapters if needed
if save_merged:
    print(f"Merging model to {merged_dir}...")
    model = trainer.model.merge_and_unload()  # Merge LoRA adapters
    model.save_pretrained(merged_dir)  # Save the merged model
    tokenizer.save_pretrained(merged_dir)  # Save the tokenizer
# delete all checkpoints after training
for item in os.listdir(output):
    item_path = os.path.join(output, item)
    if os.path.isdir(item_path) and item.startswith("checkpoint-"):
        print(f"Deleting checkpoint directory: {item_path}")
        shutil.rmtree(item_path)
