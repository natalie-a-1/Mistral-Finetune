# Mistral-7B Finetuning Template

This repository provides a template for finetuning Mistral-7B models with custom datasets. It's designed to be easy to use while providing flexibility for different data sources and formats.

## Features

- Supports multiple data sources (Hugging Face datasets, CSV, JSON, text files)
- Uses LoRA (Low-Rank Adaptation) for efficient finetuning
- 4-bit quantization for reduced memory usage
- Configurable instruction templates
- Support for combining multiple datasets
- Command-line argument support for easy configuration

## Requirements

```
transformers>=4.34.0
peft>=0.5.0
datasets>=2.14.0
bitsandbytes>=0.41.0
accelerate>=0.23.0
torch>=2.0.0
huggingface_hub>=0.17.0
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mistral-finetune-template.git
cd mistral-finetune-template
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The simplest way to use the template is to run it with command-line arguments:

```bash
python3 mistral_finetune_template.py \
  --model_id="mistralai/Mistral-7B-Instruct-v0.2" \
  --output_dir="./fine_tuned_mistral" \
  --dataset_type="csv" \
  --dataset_path="path/to/your/data.csv" \
  --dataset_text_field="text" \
  --instruction_template="Continue speaking in the style of: " \
  --num_train_epochs=3
```

### Configuration Options

The template supports many configuration options:

#### Model Parameters
- `--model_id`: The model ID to finetune (default: "mistralai/Mistral-7B-Instruct-v0.2")
- `--output_dir`: Directory to save the finetuned model (default: "./fine_tuned_mistral")

#### Hugging Face Hub Parameters
- `--use_hf_token`: Whether to use a Hugging Face token (default: False)
- `--hf_token`: Hugging Face token for accessing gated models

#### Dataset Parameters
- `--dataset_type`: Type of dataset: 'huggingface', 'csv', 'json', 'text', or 'custom' (default: "custom")
- `--dataset_path`: Path to dataset or Hugging Face dataset ID
- `--dataset_text_field`: Field containing the text in the dataset (default: "text")
- `--second_dataset_path`: Optional path to a second dataset to combine with the first
- `--second_dataset_text_field`: Field containing the text in the second dataset

#### Instruction Format
- `--instruction_template`: Template for the instruction part (default: "Continue speaking in the style of: ")

#### Training Parameters
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device during training (default: 4)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_seq_length`: Maximum sequence length for tokenization (default: 1024)

#### LoRA Parameters
- `--lora_r`: LoRA attention dimension (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 64)
- `--lora_dropout`: LoRA dropout probability (default: 0.05)

#### Advanced Parameters
- `--seed`: Random seed for reproducibility (default: 42)
- `--push_to_hub`: Whether to push the model to the Hugging Face Hub (default: False)
- `--hub_model_id`: Model ID for pushing to the Hugging Face Hub

## Examples

### Finetuning with a CSV Dataset

```bash
python mistral_finetune_template.py \
  --model_id="mistralai/Mistral-7B-Instruct-v0.2" \
  --output_dir="./fine_tuned_shakespeare" \
  --dataset_type="csv" \
  --dataset_path="shakespeare_data.csv" \
  --dataset_text_field="text" \
  --instruction_template="Continue writing in the style of Shakespeare: " \
  --num_train_epochs=3 \
  --per_device_train_batch_size=4
```

### Finetuning with a Hugging Face Dataset

```bash
python mistral_finetune_template.py \
  --model_id="mistralai/Mistral-7B-Instruct-v0.2" \
  --output_dir="./fine_tuned_code" \
  --dataset_type="huggingface" \
  --dataset_path="codeparrot/github-code" \
  --dataset_text_field="content" \
  --instruction_template="Write Python code for: " \
  --num_train_epochs=2
```

### Combining Two Datasets

```bash
python mistral_finetune_template.py \
  --model_id="mistralai/Mistral-7B-Instruct-v0.2" \
  --output_dir="./fine_tuned_combined" \
  --dataset_type="csv" \
  --dataset_path="dataset1.csv" \
  --dataset_text_field="text" \
  --second_dataset_path="dataset2.csv" \
  --second_dataset_text_field="content" \
  --instruction_template="Continue in this style: "
```

## Customizing for Your Own Data

For custom dataset loading, modify the `load_and_process_dataset` function in the script. Look for the section with:

```python
elif config.dataset_type == "custom":
    # Implement your custom dataset loading logic here
    # This is a placeholder - replace with your own logic
    logger.warning("Using custom dataset type - you need to implement your loading logic")
    dataset = Dataset.from_dict({"text": ["Example text"]})
```

Replace this with your own dataset loading logic.

## Compatible Models

While this template is specifically designed for Mistral-7B models, it can be adapted for other transformer-based models with minimal changes. Compatible models include:

- **Mistral Family**: 
  - Mistral-7B (optimized for this model)
  - Mixtral-8x7B (may require additional memory)
  
- **Llama Family**:
  - Llama 2 (7B, 13B)
  - Code Llama
  - Llama 3 (8B)
  
- **Other Models**:
  - Falcon models
  - MPT models
  - Pythia models
  - BLOOMZ models

The main adaptations needed for other models would be:

1. Changing the model ID
2. Adjusting the instruction format template to match the model's expected format
3. Modifying the target modules for LoRA depending on the model architecture

When switching to a different model, remember to check its documentation for the correct prompt/instruction format.

## Technical Details

### Why This Approach Works Well for Mistral-7B

This finetuning approach is particularly well-suited for Mistral-7B for several reasons:

1. **Instruction Format Alignment**: The template uses Mistral's specific instruction format (`<s>[INST] ... [/INST] ...`) which is crucial for maintaining the model's instruction-following capabilities.

2. **LoRA Target Modules**: The LoRA configuration targets specific attention modules in Mistral's architecture, optimized based on the model's design.

3. **Quantization Strategy**: The 4-bit quantization with nf4 datatype is particularly effective for Mistral models, offering an excellent balance between memory efficiency and performance.

4. **Training Hyperparameters**: The default learning rate, batch size, and other parameters are specifically tuned for Mistral finetuning based on empirical results.

When Mistral was originally trained, it used a specific instruction format and training approach that this template maintains. This consistency helps preserve the model's core capabilities while adapting it to your specific domain.

## Learning Resources

If you're new to model finetuning, here are some resources to help you understand the concepts and techniques used in this template:

### Introductory Resources

1. **[Parameter-Efficient Fine-Tuning (PEFT) Guide](https://huggingface.co/docs/peft/index)** - Comprehensive guide on PEFT methods like LoRA from Hugging Face

2. **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** - The original paper describing the LoRA technique

3. **[Hugging Face Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training)** - General guide on fine-tuning transformer models

### Advanced Topics

4. **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** - Paper on quantized LoRA, which this template implements

5. **[Mistral 7B Technical Documentation](https://mistral.ai/news/announcing-mistral-7b/)** - Technical details about the Mistral model architecture

6. **[Instruction Tuning for LLMs](https://arxiv.org/abs/2210.11416)** - Understanding how instruction finetuning works

### Video Tutorials

7. **[Fine-tuning LLMs for Beginners](https://www.youtube.com/watch?v=eC6Hd1hFvos)** - Practical walkthrough of the finetuning process

### Practical Guides

9. **[Hugging Face PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)** - Example scripts for various PEFT methods

10. **[LoRA Fine-tuning Explained](https://lightning.ai/pages/community/tutorial/lora-llm/)** - Step-by-step guide from Lightning AI

## Quick Start Guide

If you're a researcher looking to get started with finetuning, here's a simplified workflow:

1. **Prepare your dataset**: Format your text data as CSV, JSON, or text files. Make sure each example is well-formatted and relevant to your task.

2. **Define your instruction template**: Think about what type of behavior you want to teach the model. For example, "Respond to this medical question with accurate information:" for a medical assistant.

3. **Start with a small test run**: Begin with a small subset of your data and 1 epoch to ensure everything works.

4. **Scale up gradually**: Once your setup works, increase the dataset size and number of epochs.

5. **Evaluate carefully**: After finetuning, extensively test your model to ensure it behaves as expected and hasn't developed unwanted behaviors.

For academic research, remember to document your hyperparameters, dataset statistics, and evaluation methods to ensure reproducibility.

## Hardware Recommendations

- **Minimum**: 24GB VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **Recommended**: 40GB+ VRAM (A100, A6000, H100)
- **Cloud options**: 
  - Google Colab Pro+ with A100
  - Lambda Labs instances
  - Vast.ai GPU rentals
  - RunPod services

For limited hardware, consider:
- Reducing batch size and increasing gradient accumulation steps
- Using 8-bit quantization for the base model
- Reducing context length
- Finetuning smaller models (e.g., 7B instead of 13B+)

## Hardware Requirements

- At least 24GB VRAM is recommended for finetuning Mistral-7B with 4-bit quantization
- For smaller GPUs, you may need to reduce batch size and use gradient accumulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This template is based on best practices from the Hugging Face ecosystem
- Special thanks to the creators of Mistral-7B, PEFT, and BitsAndBytes libraries 