# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import fire
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_translation_dataset(dataset_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Loads a translation dataset from a JSON file.
    Supports both JSONL format (one JSON object per line) and JSON array format.
    Each object should have an 'instruction' field for the source text and 'output' field for the reference translation.
    
    Args:
        dataset_path: Path to the dataset file
        
    Returns:
        List of dictionaries with 'text' and 'label' keys
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        ValueError: If the dataset format is invalid
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = []
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # Try to parse as JSON array first
            try:
                data_list = json.loads(content)
                if isinstance(data_list, list):
                    for i, item in enumerate(data_list):
                        if not isinstance(item, dict):
                            logger.warning(f"Skipping non-dict item at index {i}")
                            continue
                            
                        if "instruction" in item and "input" in item:
                            # 新的格式：instruction + input 组合
                            combined_text = f"{item['instruction']}\n\n{item['input']}"
                            dataset.append({
                                "text": combined_text.strip(),
                                "label": str(item.get("output", "")).strip()
                            })
                        elif "instruction" in item:
                            # 旧格式：只有instruction
                            dataset.append({
                                "text": str(item["instruction"]).strip(),
                                "label": str(item.get("output", "")).strip()
                            })
                        elif "text" in item:
                            # 其他格式：text字段
                            dataset.append({
                                "text": str(item["text"]).strip(),
                                "label": str(item.get("label", "")).strip()
                            })
                        else:
                            logger.warning(f"Skipping item at index {i}: missing 'instruction' or 'text' field")
                    
                    logger.info(f"Loaded {len(dataset)} samples from JSON array")
                    return dataset
            except json.JSONDecodeError as e:
                logger.debug(f"Not a JSON array format: {e}")
            
            # If not a JSON array, try JSONL format
            f.seek(0)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        logger.warning(f"Skipping non-dict item at line {line_num}")
                        continue
                        
                    if "instruction" in data and "input" in data:
                        # 新的格式：instruction + input 组合
                        combined_text = f"{data['instruction']}\n\n{data['input']}"
                        dataset.append({
                            "text": combined_text.strip(),
                            "label": str(data.get("output", "")).strip()
                        })
                    elif "text" in data:
                        dataset.append({
                            "text": str(data["text"]).strip(),
                            "label": str(data.get("label", "")).strip()
                        })
                    elif "instruction" in data:
                        # 旧格式：只有instruction
                        dataset.append({
                            "text": str(data["instruction"]).strip(),
                            "label": str(data.get("output", "")).strip()
                        })
                    else:
                        logger.warning(f"Skipping line {line_num}: missing 'text' or 'instruction' field")
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
            
            logger.info(f"Loaded {len(dataset)} samples from JSONL format")
            return dataset
            
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {dataset_path}: {e}")

def translate_with_vllm(
    model_name_or_path: str,
    dataset_path: str,
    source_lang: str = "auto",
    target_lang: str = "English",
    save_name: str = "translated_predictions.jsonl",
    prompt_template: str = "### Instruction:\nPlease choose the correct answer from the given options. Answer with the number: first answer = 0, second answer = 1\n\n### Input:\n{text}\n\n### Response:\n",
    temperature: float = 0.95,
    top_p: float = 0.7,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    batch_size: int = 256,
    tensor_parallel_size: Optional[int] = None,
    adapter_name_or_path: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
):
    """
    Performs batch translation using the vLLM engine.

    Args:
        model_name_or_path: The model to use for translation.
        dataset_path: Path to the JSON Lines dataset file.
        source_lang: The source language (unused in current implementation).
        target_lang: The target language (unused in current implementation).
        save_name: Output file name for the generated translations.
        prompt_template: Template for the translation prompt. Use {text} as placeholder.
        temperature: Sampling temperature (0.0 to 1.0).
        top_p: Nucleus sampling parameter (0.0 to 1.0).
        max_new_tokens: Maximum number of tokens to generate.
        repetition_penalty: Penalty for repetition (1.0 = no penalty).
        skip_special_tokens: Whether to skip special tokens in the output.
        batch_size: The number of examples to process in each batch.
        tensor_parallel_size: The number of GPUs to use for tensor parallelism.
        adapter_name_or_path: Path to the LoRA adapter.
        gpu_memory_utilization: Fraction of GPU memory to use.
        max_model_len: Maximum sequence length for the model.
        trust_remote_code: Whether to trust remote code in the model.
        dtype: Data type for the model ('auto', 'float16', 'bfloat16').
    """

    # 1. Initialize vLLM engine and sampling parameters
    logger.info("Initializing vLLM engine...")
    
    # Prepare engine arguments
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
        "enable_lora": adapter_name_or_path is not None,
        "max_lora_rank": 128,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    
    # Add optional parameters
    if tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = tensor_parallel_size
    if max_model_len is not None:
        engine_args["max_model_len"] = max_model_len
    
    try:
        llm = LLM(**engine_args)
        logger.info("vLLM engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        raise

    # Initialize sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        skip_special_tokens=skip_special_tokens,
    )
    
    # Initialize LoRA request if adapter is provided
    lora_request = None
    if adapter_name_or_path:
        if not os.path.exists(adapter_name_or_path):
            logger.warning(f"LoRA adapter path does not exist: {adapter_name_or_path}")
        else:
            lora_request = LoRARequest("default", 1, adapter_name_or_path)
            logger.info(f"Using LoRA adapter: {adapter_name_or_path}")

    # 2. Load the dataset
    try:
        dataset = load_translation_dataset(dataset_path)
        if not dataset:
            logger.warning("Dataset is empty. Exiting.")
            return
        logger.info(f"Successfully loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # 3. Process batches and perform inference
    all_results = []
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    logger.info(f"Starting translation with {num_batches} batches of size {batch_size}")
    
    # Statistics tracking
    successful_batches = 0
    failed_samples = 0
    
    try:
        for i in tqdm(range(0, num_samples, batch_size), desc="Translating in batches", unit="batch"):
            batch = dataset[i : min(i + batch_size, num_samples)]
            
            # Prepare prompts for the current batch
            prompts = []
            for item in batch:
                try:
                    prompt = prompt_template.format(text=item["text"])
                    prompts.append(prompt)
                except KeyError as e:
                    logger.warning(f"Skipping item missing required field: {e}")
                    continue
            
            if not prompts:
                logger.warning(f"Skipping empty batch at index {i}")
                continue

            # Generate translations
            try:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
                successful_batches += 1
                
                # Collect results
                for item, output in zip(batch, outputs):
                    if output.outputs:
                        predicted_text = output.outputs[0].text.strip()
                        result = {
                            "prompt": item.get("text", ""),  # 添加原文
                            "predict": predicted_text,
                            "label": item.get("label", "")
                        }
                        all_results.append(result)
                    else:
                        logger.warning("Empty output from model")
                        all_results.append({
                            "prompt": item.get("text", ""),  # 添加原文
                            "predict": "",
                            "label": item.get("label", "")
                        })
                        failed_samples += 1
                        
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                # Add empty results for failed batch
                for item in batch:
                    all_results.append({
                        "prompt": item.get("text", ""),  # 添加原文
                        "predict": "",
                        "label": item.get("label", "")
                    })
                    failed_samples += 1
                continue

            # Memory cleanup
            if i % (batch_size * 10) == 0:  # Every 10 batches
                gc.collect()
                
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        return
    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}")
        raise

    # 4. Save results to a file
    save_path = Path(save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(all_results)} translated results to {save_path}...")
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        logger.info("=" * 50)
        logger.info(f"✅ Translation completed successfully!")
        logger.info(f"📁 Results saved to: {save_path.absolute()}")
        logger.info(f"📊 Total samples processed: {len(all_results)}")
        logger.info(f"✅ Successful batches: {successful_batches}/{num_batches}")
        logger.info(f"❌ Failed samples: {failed_samples}")
        logger.info(f"💾 File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
    
    # Final cleanup
    del llm
    gc.collect()

if __name__ == "__main__":
    fire.Fire(translate_with_vllm)