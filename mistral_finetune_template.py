#!/usr/bin/env python3
"""
Mistral-7B Finetuning Template

This script provides a template for finetuning Mistral-7B models with custom datasets.
It supports various data sources and formats, and uses LoRA for efficient finetuning.

Usage:
    1. Configure the parameters in the CONFIG section
    2. Prepare your dataset(s) according to the format requirements
    3. Run the script: python mistral_finetune_template.py
"""

import os
import torch
import logging
from typing import Dict, Sequence, List, Optional, Union
from dataclasses import dataclass, field

# Required libraries
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, concatenate_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#############################################
# CONFIGURATION - MODIFY THESE PARAMETERS
#############################################

@dataclass
class FinetuningConfig:
    """Configuration for the finetuning process"""
    # Model parameters
    model_id: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={"help": "The model ID to finetune"}
    )
    output_dir: str = field(
        default="./fine_tuned_mistral",
        metadata={"help": "Directory to save the finetuned model"}
    )
    
    # Hugging Face Hub parameters
    use_hf_token: bool = field(
        default=False,
        metadata={"help": "Whether to use a Hugging Face token for accessing gated models"}
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face token for accessing gated models"}
    )
    
    # Dataset parameters
    dataset_type: str = field(
        default="custom",
        metadata={"help": "Type of dataset: 'huggingface', 'csv', 'json', 'text', or 'custom'"}
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to dataset or Hugging Face dataset ID"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Field containing the text in the dataset"}
    )
    second_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a second dataset to combine with the first"}
    )
    second_dataset_text_field: Optional[str] = field(
        default=None,
        metadata={"help": "Field containing the text in the second dataset"}
    )
    
    # Instruction format
    instruction_template: str = field(
        default="Continue speaking in the style of: ",
        metadata={"help": "Template for the instruction part"}
    )
    
    # Training parameters
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per device during training"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=2e-4,
        metadata={"help": "Learning rate"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    
    # LoRA parameters
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    
    # Advanced parameters
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hugging Face Hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID for pushing to the Hugging Face Hub"}
    )

def main():
    """Main function to run the finetuning process"""
    # Parse arguments
    parser = HfArgumentParser([FinetuningConfig])
    config = parser.parse_args_into_dataclasses()[0]
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Step 1: Login to Hugging Face if token is provided
    if config.use_hf_token and config.hf_token:
        logger.info("Logging in to Hugging Face Hub")
        login(token=config.hf_token)
    
    # Step 2: Configure quantization for memory-efficient loading
    logger.info("Configuring quantization")
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Step 3: Load the model and tokenizer
    logger.info(f"Loading model and tokenizer: {config.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, 
            padding_side="right",
            use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}")
        raise
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Step 4: Prepare the model for training
    logger.info("Preparing model for training")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    
    # Step 5: Apply LoRA for fine-tuning
    logger.info("Applying LoRA configuration")
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Step 6: Load and process the dataset(s)
    logger.info("Loading and processing dataset(s)")
    
    def format_instruction(text: str) -> str:
        """Format text with instruction template"""
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            return ""
        return f"<s>[INST] {config.instruction_template} [/INST] {text}</s>"
    
    def load_and_process_dataset(path: str, text_field: str) -> Dataset:
        """Load and process a dataset from various sources"""
        if not path:
            return None
            
        # Load dataset based on type
        if config.dataset_type == "huggingface":
            # Load from Hugging Face Datasets
            dataset = load_dataset(path)
            # Get the first split (usually 'train')
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
        elif config.dataset_type == "csv":
            # Load from CSV file
            import pandas as pd
            df = pd.read_csv(path)
            dataset = Dataset.from_pandas(df)
        elif config.dataset_type == "json":
            # Load from JSON file
            dataset = Dataset.from_json(path)
        elif config.dataset_type == "text":
            # Load from text file (one example per line)
            with open(path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            dataset = Dataset.from_dict({"text": texts})
        elif config.dataset_type == "custom":
            # Implement your custom dataset loading logic here
            # This is a placeholder - replace with your own logic
            logger.warning("Using custom dataset type - you need to implement your loading logic")
            dataset = Dataset.from_dict({"text": ["Example text"]})
        else:
            raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
        
        # Process the dataset
        def process_example(example):
            text = example.get(text_field, "")
            return {"text": format_instruction(text)}
        
        processed = dataset.map(
            process_example,
            desc="Processing dataset"
        )
        
        # Filter out empty examples
        processed = processed.filter(
            lambda x: len(x["text"]) > 0,
            desc="Filtering valid examples"
        )
        
        return processed
    
    # Load primary dataset
    primary_dataset = load_and_process_dataset(
        config.dataset_path, 
        config.dataset_text_field
    )
    
    # Load secondary dataset if specified
    secondary_dataset = None
    if config.second_dataset_path and config.second_dataset_text_field:
        secondary_dataset = load_and_process_dataset(
            config.second_dataset_path,
            config.second_dataset_text_field
        )
    
    # Merge datasets if needed
    if secondary_dataset:
        logger.info("Merging datasets")
        merged_dataset = concatenate_datasets([primary_dataset, secondary_dataset])
    else:
        merged_dataset = primary_dataset
    
    # Step 7: Split dataset into training and evaluation sets
    logger.info("Splitting dataset into train and evaluation sets")
    dataset = merged_dataset.train_test_split(test_size=0.1, seed=config.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Evaluation examples: {len(eval_dataset)}")
    
    # Step 8: Tokenize the datasets
    logger.info("Tokenizing datasets")
    def tokenize_function(examples: Dict[str, Sequence[str]]) -> dict:
        """Tokenize with proper labels for causal language modeling"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    
    tokenized_eval = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    # Step 9: Configure data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # Step 10: Define training arguments
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=100,
        logging_steps=25,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        report_to="none",
        gradient_checkpointing=True,
        save_total_limit=3,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=config.seed,
    )
    
    # Step 11: Set up training
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Step 12: Train the model
    logger.info("Starting training")
    try:
        model.train()
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    
    # Step 13: Save the fine-tuned model
    logger.info(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 