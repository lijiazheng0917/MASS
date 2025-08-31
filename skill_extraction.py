"""
Script for labeling mathematical benchmark datasets with knowledge points.

This script uses a large language model to analyze mathematical questions and 
identify relevant knowledge points. The output is saved in JSONL format.
"""

import json
import re
import argparse
from typing import Dict, List, Optional, Any

from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


# Configuration
DEFAULT_MODEL_NAME = './Qwen2.5-72B-Instruct-GPTQ-Int4'
DEFAULT_DATA_DIR = './numina_100k.jsonl'
DEFAULT_OUTPUT_DIR = './numina_100k_skills.jsonl'


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Label mathematical benchmark dataset with knowledge points.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="Path to the language model"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=DEFAULT_DATA_DIR,
        help="Path to the input dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help="Path to save the labeled output"
    )
    return parser.parse_args()


def generate_prompt(text: str) -> str:
    """
    Generate a prompt for the language model to analyze mathematical content.
    
    Args:
        text (str): The mathematical question or content to analyze
        
    Returns:
        str: Formatted prompt for the language model
    """
    content = f'''
Please assume the role of a math teacher and analyze the provided question with the following steps:

1. Determine if the text involves mathematical knowledge, reasoning, or problem-solving skills. Respond with "YES" or "NO".
2. Identify 1-10 concise, general mathematical knowledge points being tested. 

Note: 
1. Ensure each knowledge point is abstract and generalized. 
2. Avoid using verbs; focus on noun phrases (e.g., "function symmetry" instead of "analyzing function symmetry").
3. Avoid specifics like numbers, angles, or exact values. Focus on overarching concepts and techniques.
4. Keep each knowledge point to a maximum of 10 words.

Provide your assessment in one JSON format:

```json
{{
  "math relevance": "YES" or "NO",
  "knowledge points": ["point1", "point2", ...]
}}
```

Provided text:
{text}

JSON output:
'''

    return content


def extract_json(content: str) -> str:
    """
    Extract JSON content from model response.
    
    Args:
        content (str): Raw model response containing JSON
        
    Returns:
        str: Extracted JSON string
    """
    # Pattern to match JSON wrapped in triple backticks
    pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, content, re.DOTALL)
    if len(matches) > 0:
        content = matches[0]
    
    content = content.replace('\\"', "'")
    content = content.replace("```json", "")
    content = content.replace("```", "").strip()

    return content


def process_model_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Process model response and convert to structured data.
    
    Args:
        response (str): Raw model response
        
    Returns:
        Optional[Dict[str, Any]]: Parsed JSON data or None if parsing fails
    """
    try:
        json_str = extract_json(response)
        result = json.loads(json_str)
        return result
    except Exception:
        return None


def main():
    """Main function to run the labeling process."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load dataset
    dataset = load_dataset('json', data_files={'train': args.data_dir})
    
    # Generate prompts for all data points
    prompt_list = []
    for data in dataset['train']:
        prompt = generate_prompt(data['content'])
        prompt_list.append(prompt)

    # Initialize model and generation parameters
    llm = LLM(model=args.model_name)
    sampling_params = SamplingParams(temperature=0, max_tokens=300)

    # Generate responses
    outputs = llm.generate(prompt_list, sampling_params)

    # Process and save results
    with open(args.output_dir, 'w', encoding='utf-8') as outfile:
        for i, output in tqdm(enumerate(outputs)):  
            prompt = output.prompt
            generated_text = output.outputs[0].text
            extracted_json = process_model_response(generated_text)
            if extracted_json is not None:
                extracted_json['content'] = dataset['train'][i]['content']
                json_line = json.dumps(extracted_json, ensure_ascii=False)
                outfile.write(json_line + '\n')


if __name__ == "__main__":
    main()
