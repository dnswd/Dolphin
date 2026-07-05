""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os
import sys
import hashlib

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.utils import *

# Force line-buffered stdout so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Set device and precision
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.bfloat16
        else:
            self.device = "cpu"
            dtype = torch.float32

        # Load model from local path or Hugging Face hub with memory-efficient dtype
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype
        )
        self.model.eval()
        self.model.to(self.device)
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

    def chat(self, prompt, image):
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        assert len(images) == len(prompts)
        
        # preprocess all images
        processed_images = [resize_img(img) for img in images]
        # generate all messages
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)

        # prepare all texts
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        # collect all image inputs
        all_image_inputs = []
        all_video_inputs = None
        for msgs in all_messages:
            image_inputs, video_inputs = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        # prepare model inputs
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # inference
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=None,
                use_cache=True,
                # repetition_penalty=1.05
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        results = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Free VRAM between batches to avoid fragmentation on smaller GPUs
        del inputs, generated_ids, generated_ids_trimmed
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results


def process_document(document_path, model, save_dir, max_batch_size=None, post_process=False):
    """Parse documents with two stages - Handles both images and PDFs"""
    # TODO: Save to folder with digest
    # TODO: Page checkpoint handling
    file_ext = os.path.splitext(document_path)[1].lower()
    
    if file_ext == '.pdf':
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")
        
        all_results = []
        
        # Process each page
        for page_idx, pil_image in enumerate(images):
            # Skip blank pages (std dev of grayscale pixel values near zero)
            gray = np.array(pil_image.convert("L"), dtype=np.float32)
            if gray.std() < 8.0:
                print(f"Skipping page {page_idx + 1}/{len(images)} (blank)", flush=True)
                continue

            print(f"Processing page {page_idx + 1}/{len(images)}", flush=True)
            
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # Process this page (don't save individual page results)
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False, post_process=post_process
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)
        
        # Save combined results for multi-page PDF
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir, post_process=post_process)
        
        return combined_json_path, all_results
    
    else:
        # Process regular image file
        pil_image = Image.open(document_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size, post_process=post_process)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True, post_process=False):
    """Process a single image (either from file or converted from PDF page)
    
    Args:
        image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output file
        max_batch_size: Maximum batch size for processing
        save_individual: Whether to save individual results (False for PDF pages)
        
    Returns:
        Tuple of (json_path, recognition_results)
    """
    # Stage 1: Page-level layout and reading order parsing
    layout_output = model.chat("Parse the reading order of this document.", image)
    # print(layout_output)

    # Stage 2: Element-level content parsing
    recognition_results = process_elements(layout_output, image, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested (skip for PDF pages)
    json_path = None
    if save_individual:
        # Create a dummy image path for save_outputs function
        json_path = save_outputs(recognition_results, image, image_name, save_dir, post_process=post_process)

    return json_path, recognition_results


def process_elements(layout_results, image, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
    # Check for bbox overlap - if too many overlaps, treat as distorted page
    elif len(layout_results_list) > 1 and check_bbox_overlap(layout_results_list, image):
        print("Falling back to distorted_page mode due to high bbox overlap", flush=True)
        layout_results_list = [([0, 0, *image.size], 'distorted_page', [])]
        
    tab_elements = []      
    equ_elements = []     
    code_elements = []    
    text_elements = []     
    figure_results = []    
    reading_order = 0

    # Collect elements and group
    for bbox, label, tags in layout_results_list:
        try:
            if label == "distorted_page":
                x1, y1, x2, y2 = 0, 0, *image.size
                pil_crop = image
            else:
                # get coordinates in the original image
                x1, y1, x2, y2 = process_coordinates(bbox, image)
                # crop the image
                pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:
                if label == "fig":
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:
                    # Prepare element information
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    }
                    
                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}", flush=True)
            continue

    recognition_results = figure_results.copy()
    
    if tab_elements:
        results = process_element_batch(tab_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if equ_elements:
        results = process_element_batch(equ_elements, model, "Read formula in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if code_elements:
        results = process_element_batch(code_elements, model, "Read code in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if text_elements:
        results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []
    
    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)
        
        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)
        
        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
                "tags": elem["tags"],
            })
    
    return results

def md5_digest(filepath, chunk_size=8192):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_sources(source_folder, file_extensions):
    """Collect and calculate all source hashes from the source_folder with file_extensions"""
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder {source_folder} does not exist")
    if not os.path.isdir(source_folder):
        raise NotADirectoryError(f"Source path {source_folder} is not a directory")
 
    matched_files = []
    for ext in file_extensions:
        matched_files.extend(glob.glob(os.path.join(source_folder, f"*{ext}")))
    matched_files = sorted(set(matched_files))
 
    return [(filepath, md5_digest(filepath)) for filepath in matched_files]

def resolve_save_paths(save_dir, filepath, digest):
    filename = os.path.basename(filepath)
    file_save = f"{filename}_{digest}"
    return os.path.join(save_dir, file_save)

def main():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, default="./source", help="Path to input image/PDF or directory of files")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./parsed_source",
        help="Directory to save parsing results (default: ./parsed_source)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=4,
        help="Maximum number of document elements to parse in a single batch (default: 4)",
    )
    parser.add_argument("--post_process", action="store_true", help="Whether to apply post-processing to the output results")
    args = parser.parse_args()

    # Load Model
    model = DOLPHIN(args.model_path)

    # Collect Document Files (images and PDFs)
    document_files = collect_sources(args.input_path, [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"])

    # Ensure save dir exists
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = args.save_dir


    total_samples = len(document_files)
    print(f"\nTotal files to process: {total_samples}", flush=True)

    # Process All Document Files
    for file_path, digest in document_files:
        print(f"\nProcessing {file_path}", flush=True)
        file_save_dir = resolve_save_paths(save_dir, file_path, digest)
        setup_output_dirs(file_save_dir)
        try:
            json_path, recognition_results = process_document(
                document_path=file_path,
                model=model,
                save_dir=file_save_dir,
                max_batch_size=args.max_batch_size,
                post_process=args.post_process
            )

            print(f"Processing completed. Results saved to {save_dir}", flush=True)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", flush=True)
            continue


if __name__ == "__main__":
    main()
