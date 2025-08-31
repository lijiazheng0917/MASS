"""
MASS Trainer Script

This script provides training functionality for the MASS model using Hugging Face Transformers.
It supports training with configurable parameters, dataset selection, and deepspeed optimization.

Key Features:
- Configurable training parameters via command line arguments
- Dataset selection with ratio-based sampling
- Support for checkpoint resumption
- DeepSpeed integration for memory-efficient training
- Flash attention for improved performance
"""

import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def parse_arguments():
    """
    Parse command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MASS Trainer")
    parser.add_argument("--model_name", type=str, default='./TinyLlama-1.1B-intermediate-step-1431k-3T/',
                        help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, default='./full_numina_tinyllama',
                        help="Directory to save model checkpoints and outputs")
    parser.add_argument("--data_dir", type=str, default='./full_numina_tinyllama/',
                        help="Directory containing tokenized training data")
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint file for resuming training', default=None)
    parser.add_argument("--ratio", type=float, default=1,
                        help="Ratio of dataset to use for training (0.0 to 1.0)")
    parser.add_argument("--epochs", type=float, default=1,
                        help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Number of steps between checkpoint saves")
    parser.add_argument("--grad_acc", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for training")
    parser.add_argument("--bs", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--grad_ckpt", type=bool, default=False,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--use_cache", type=bool, default=False,
                        help="Enable model cache for faster inference")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, use_cache=False):
    """
    Load the pre-trained model and tokenizer.
    
    Args:
        model_name (str): Path to the pre-trained model
        use_cache (bool): Whether to enable model cache
        
    Returns:
        tuple: (tokenizer, model)
    """
    # Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with bfloat16 precision and flash attention for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    model.config.use_cache = use_cache
    
    return tokenizer, model


def load_and_prepare_dataset(data_dir, ratio=1.0):
    """
    Load and prepare the training dataset with optional sampling.
    
    Args:
        data_dir (str): Directory containing tokenized datasets
        ratio (float): Ratio of dataset to use (0.0 to 1.0)
        
    Returns:
        Dataset: Prepared training dataset
    """
    # Load tokenized datasets from disk
    tokenized_datasets = load_from_disk(data_dir)
    print(f"Loaded dataset: {tokenized_datasets}")
    
    # Select subset of data based on ratio
    total_size = len(tokenized_datasets['train'])
    select_size = int(total_size * ratio)
    selected_data = tokenized_datasets['train'].select(range(select_size))
    print(f"Selected {select_size} samples from the dataset (total: {total_size}).")
    
    return selected_data


def main():
    """
    Main training function that orchestrates the entire training process.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.use_cache)
    
    # Calculate and display model size
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    selected_data = load_and_prepare_dataset(args.data_dir, args.ratio)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Create training arguments
    print("Setting up training arguments...")
    train_args = TrainingArguments( 
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bs,
        # per_device_eval_batch_size=8,
        # eval_strategy="steps",
        save_total_limit=20,     
        # load_best_model_at_end=True,
        # eval_steps=1000,
        logging_steps=10,
        gradient_accumulation_steps=args.grad_acc,
        max_grad_norm= 1,
        num_train_epochs=args.epochs,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        save_steps=args.save_steps,
        bf16=True,
        deepspeed='ds.json',
        gradient_checkpointing=args.grad_ckpt,
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=selected_data,
    )
    
    # Start training
    print("Starting training...")
    if args.ckpt:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
