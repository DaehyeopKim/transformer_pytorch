## 1. Load Context and Query
## 2. Tokenize Context and Query
## 3. Load Model
## 4. Encode Context and Query
## 5. Inference (Searching / Sampling)

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
    parser.add_argument("--context", type=str, required=True, help="Context input file path")
    parser.add_argument("--query", type=str, required=True, help="Query input file path")
    parser.add_argument("--log", type=str, default="info", choices=["debug", "info", "no"],
                        help="Set logging level (no : warning, error, critical). ")
    parser.add_argument("--infermode", type=str, default="greedy", 
                        choices=["greedy", "beam", "top_k", "top_p"], 
                        help="Set inference mode")
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to the model file")
    
    args = parser.parse_args()

    infer_logger = logging.getLogger("inference.py")

    setup_logging(args.log, infer_logger)
    infer_logger.info("Logging is set to %s", args.log)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load context and query
    context = context
    query = args.query

    # Tokenize context and query
    tokenizer = BytePairEncoder(device=device)   
    context_tokens_ids = tokenizer.encode(context) #torch.Tensor
    query_tokens_ids = tokenizer.encode(query) #torch.Tensor
    
    # Prepare input token IDs
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id
    input_token_ids = torch.cat((torch.tensor([bos_token_id]), context_tokens_ids, torch.tensor([sep_token_id]), query_tokens_ids), dim=0)


    # Load the model
    model = load_transformer_with_config(args.model_path, device)
    
    # Perform inference (searching/sampling)
    results_tokens = model.inference(input_token_ids.unsqueeze(0).to(device), inference_method = args.infermode)
    results = tokenizer.decode(results_tokens)

    print("Inference results:\n", results)

if __name__ == "__main__":
    main()