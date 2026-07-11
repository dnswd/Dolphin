import os
import glob
import torch
import hashlib
import sqlite3
import threading

from pathlib import Path
from argparse import ArgumentParser
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from utils.utils import *


# Force line-buffered stdout so progress is visible in real time
sys.stdout.reconfigure(line_buffering=True)

FILE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
DDL = # sql
'''
PRAGMA foreign_keys = ON; -- enforce foreign key constraints
CREATE TABLE IF NOT EXISTS documents (
    hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pages (
    doc_hash TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    -- TODO: steps
    
    PRIMARY KEY (doc_hash, page_number),
    FOREIGN KEY (doc_hash) REFERENCES documents (hash) ON DELETE CASCADE
);
'''


def init_serve():
    raise NotImplementedError("Not yet implemented")  
    
def md5_digest(filepath, chunk_size=8192):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_sources(source):
    """Collect and calculate all source hashes from the source_folder with file_extensions"""
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path {source} does not exist")
    
    if os.path.isdir(source):
        matched_files = []
        for ext in FILE_EXTENSIONS:
            # only check 1 level
            matched_files.extend(glob.glob(os.path.join(source, f"*{ext}")))
        matched_files = sorted(set(matched_files))
     
        return [(filepath, md5_digest(filepath)) for filepath in matched_files]
    
    if os.path.isfile(source):
        return [(source, md5_digest(source))]

    raise FileNotFoundError(f"Unable to determine if {source} is a path or directory")

def configure_document_output_dir(path, hash, output_path):
    filename = os.path.basename(path)
    foldername = f"{filename}_{hash}"
    output_folder = os.path.join(output_path, foldername)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def process_element_batch(elements, model, prompt, max_batch_size=4):
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

def process_elements(layout_results, image, model, save_dir, image_name):
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
        results = process_element_batch(tab_elements, model, "Parse the table in the image.")
        recognition_results.extend(results)
    
    if equ_elements:
        results = process_element_batch(equ_elements, model, "Read formula in the image.")
        recognition_results.extend(results)
    
    if code_elements:
        results = process_element_batch(code_elements, model, "Read code in the image.")
        recognition_results.extend(results)
    
    if text_elements:
        results = process_element_batch(text_elements, model, "Read text in the image.")
        recognition_results.extend(results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results

def process_image_document(model, path, hash, output_dir):
    pil = Image.open(path).convert("RGB")

    # layout calculation
    layout_output = model.chat("Parse the reading order of this document.", pil)

    # parse elements
    parsed_elements = process_elements(layout_output, pil, model, output_dir, os.splitext(path)[0])
    

def process_document(model, path, hash, output_path):
    output_dir = configure_document_output_dir(path, hash, output_path)
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext == '.pdf':
        return
    else: # image
        return

def init_local(model_path, source_path, output_path):
    documents = collect_sources(source_path)
    paths = [os.path.abspath(doc[0]) for doc in documents]
    width = max(len(p) for p in paths) if paths else 0

    print(f"Found {len(documents)} from input:", flush=True)
    for path, (_, md5) in zip(paths, documents):
        print(f"{path:<{width}}  (md5:{md5})", flush=True)

    print(f"\nLoading model from {os.path.abspath(model_path)}\n", flush=True)
    model = DOLPHIN(model_path)
    print(f"\nModel loaded successfully\n", flush=True)

    for (path, hash) in documents:
        process_document(model, path, hash, output_path)

    return True

class DB:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_name="dolphin_job.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DB, cls).__new__(cls)
                cls._instance._init_connection(db_name)
        return cls._instance
    
    def _init_connection(self, db_name):
        self.connection = sqlite3.connect(db_name, check_same_thread=False) # share connection between threads
        self.cursor = self.connection.cursor()
        self.db_name = db_name
        print(f"Opening database: {db_name}", flush=True)

        try:
            self.cursor.executescript(DDL)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Databse failed to apply DDL: {e}")
            self.close()
            raise

    def execute(self, query, params=()):
        with self._lock:
            try:
                self.cursor.execute(query, params)
                self.connection.commit()
                return self.cursor
            except sqlite3.Error as e:
                self.connection.rollback()
                print(f"Database error: {e}", flush=True)
                raise

    def fetch_all(self, query, params=()):
        with self._lock:
            try:
                self.cursor.execute(query, params)
                return self.cursor.fetchall()
            except sqlite3.Error as e:
                print(f"Database error: {e}", flush=True)
                raise

    def close(self):
        with self._lock:
            if self.connection:
                self.connection.close()
                print("Database connection closed", flush=True)
            if os.path.exists(self.db_name):
                try:
                    os.remove(self.db_name)
                except OSError as e:
                    print(f"Unable to clean up database: {e}")
            DB._instance = None

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

def main():
    parser = ArgumentParser(description="Resumable document parsing based on DOLPHIN")
    sub = parser.add_subparsers(dest="command", required=True)

    common = ArgumentParser(add_help=False)
    common.add_argument("--model_path", type=Path, default=Path("./hf_model"), help="Path to Hugging Face model")

    local = sub.add_parser("local", parents=[common])
    local.add_argument("path", type=Path, nargs="?", default=Path("./source"), help="Path to file/folder with files in it")
    local.add_argument("--output", type=Path, nargs="?", default=Path("./out"), help="Path to folder to output")

    serve = sub.add_parser("serve", parents=[common])
    serve.add_argument("--host", default="0.0.0.0", help="Default to 0.0.0.0")
    serve.add_argument("--port", type=int, default=8181, help="Default to 8181")

    # init db
    db = DB()
    try:
        args = parser.parse_args()
        os.makedirs(args.output, exist_ok=True)
        if (args.command == "serve"):
            return init_serve()
        else:
            return init_local(args.model_path, args.path, args.output)
    finally:
        db.close()


if __name__ == '__main__':
    print('''
You are running DOLPHIN with resumable checkpoints.
    ''')
    main()
