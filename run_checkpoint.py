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

def init_local(model_path, source_path):
    documents = collect_sources(source_path)
    paths = [os.path.abspath(doc[0]) for doc in documents]
    width = max(len(p) for p in paths) if paths else 0

    print(f"Found {len(documents)} from input:", flush=True)
    for path, (_, md5) in zip(paths, documents):
        print(f"{path:<{width}}  (md5:{md5})", flush=True)

    print(f"\nLoading model from {os.path.abspath(model_path)}\n")
    model = DOLPHIN(model_path)
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

    serve = sub.add_parser("serve", parents=[common])
    serve.add_argument("--host", default="0.0.0.0", help="Default to 0.0.0.0")
    serve.add_argument("--port", type=int, default=8181, help="Default to 8181")

    # init db
    db = DB()
    try:
        args = parser.parse_args()
        if (args.command == "serve"):
            return init_serve()
        else:
            return init_local(args.model_path, args.path)
    finally:
        db.close()


if __name__ == '__main__':
    print('''
You are running DOLPHIN with resumable checkpoints.
    ''')
    main()
