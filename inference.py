## 1. Load Context and Query (batch, each raw-untokenized sequnece length)
## 2. Tokenize Context and Query (batch, each raw-tokenized sequnece length, vocab_size)
## 3. Padding
## 5. Load Model
## 6. Encode Context and Query
## 7. Inference (Searching / Sampling)

import argparse
import logging
import torch
import models.transformer as transformer
import models.tokenizer as tokenizer
from models import Transformer, BytePairEncoder
from utils.utils import load_transformer_with_config

def setup_logging(log_level: str, infer_logger : logging.Logger, file_name: str = "inference.log"):
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "no": logging.WARNING,
    }
    file_level = level_map.get(log_level, logging.INFO)
    console_level = logging.INFO if log_level != "no" else logging.WARNING

    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    loggers = [transformer.logger, tokenizer.logger, logging.getLogger("inference.py")]

    for logger in loggers:
        logger.setLevel(logging.DEBUG) 
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

def main():
    parser = argparse.ArgumentParser(description="Inference argument parser")
    parser.add_argument("--context", type=str, nargs='+', required=True, 
                        help="Context input file path(s). Can specify multiple files.")
    parser.add_argument("--query", type=str, nargs='+', required=True, 
                        help="Query input file path(s). Can specify multiple files.")
    parser.add_argument("--log", type=str, default="info", choices=["debug", "info", "no"],
                        help="Set logging level (no : warning, error, critical). ")
    parser.add_argument("--infermode", type=str, default="greedy", 
                        choices=["greedy", "beam", "top_k", "top_p"], 
                        help="Set inference mode")
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to the model file")
    parser.add_argument("--max_length", type=int, default=250,
                        help="Maximum length of generated sequence")
    
    args = parser.parse_args()

    infer_logger = logging.getLogger("inference.py")

    setup_logging(args.log, infer_logger)
    infer_logger.info("Logging is set to %s", args.log)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_logger.info(f"Using device: {device}")

    # Validate input arguments
    if len(args.context) != len(args.query):
        infer_logger.error("Number of context files must match number of query files")
        raise ValueError("Number of context files must match number of query files")
    
    batch_size = len(args.context)
    infer_logger.info(f"Processing {batch_size} context-query pairs")

    # Load context and query files
    contexts = []
    queries = []
    
    for i, (context_file, query_file) in enumerate(zip(args.context, args.query)):
        infer_logger.info(f"Loading pair {i+1}/{batch_size}: {context_file}, {query_file}")
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read().strip()
            contexts.append(context)
            
            with open(query_file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            queries.append(query)
            
            infer_logger.info(f"Context {i+1}: {context[:100]}...")  # Log first 100 chars
            infer_logger.info(f"Query {i+1}: {query}")
            
        except FileNotFoundError as e:
            infer_logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            infer_logger.error(f"Error reading files: {e}")
            raise

    # Initialize tokenizer
    tokenizer = BytePairEncoder(device=device)   
    vocab_size = tokenizer.vocab_size
    
    # Tokenize all context and query pairs
    input_token_ids_list = []
    max_input_length = 0
    
    for context, query in zip(contexts, queries):
        context_tokens_ids = tokenizer.encode(context)
        query_tokens_ids = tokenizer.encode(query)
        
        # Prepare input token IDs for this pair
        bos_token_id = tokenizer.bos_token_id
        sep_token_id = tokenizer.sep_token_id
        
        input_token_ids = torch.cat((
            torch.tensor([bos_token_id], device=device), 
            context_tokens_ids, 
            torch.tensor([sep_token_id], device=device), 
            query_tokens_ids
        ), dim=0)
        
        input_token_ids_list.append(input_token_ids)
        max_input_length = max(max_input_length, len(input_token_ids))
    
    # Pad sequences to same length for batch processing
    padded_input_ids = []
    pad_token_id = tokenizer.pad_token_id
    
    for input_ids in input_token_ids_list:
        padding_length = max_input_length - len(input_ids)
        if padding_length > 0:
            padded_ids = torch.cat([
                input_ids, 
                torch.full((padding_length,), pad_token_id, device=device)
            ])
        else:
            padded_ids = input_ids
        padded_input_ids.append(padded_ids)
    
    # Stack into batch tensor
    batch_input_ids = torch.stack(padded_input_ids)  # Shape: (batch_size, max_seq_len)
    
    infer_logger.info(f"Batch input shape: {batch_input_ids.shape}")
    infer_logger.info(f"Max input length: {max_input_length}")

    # Load the model
    infer_logger.info(f"Loading model from {args.model_path}")
    model = load_transformer_with_config(args.model_path, device)
    model.eval()  # Set to evaluation mode
    
    # Perform batch inference
    infer_logger.info(f"Starting inference with mode: {args.infermode}")
    with torch.no_grad():
        results_tokens = model.inference(
            batch_input_ids, 
            max_length=args.max_length, 
            inference_method=args.infermode
        )
    
    infer_logger.info(f"Inference completed. Output shape: {results_tokens.shape}")
    
    # Decode results
    results = tokenizer.decode(results_tokens)
    
    # Print results
    print("=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    
    if isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Context file: {args.context[i]}")
            print(f"Query file: {args.query[i]}")
            print(f"Generated text: {result}")
            print("-" * 40)
    else:
        print(f"Generated text: {results}")
    
    infer_logger.info("Inference completed successfully")

if __name__ == "__main__":
    main()