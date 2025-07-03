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

    # y : enable logging, n : disable logging
    parser.add_argument("--log", type=str, default="info", choices=["debug", "info", "no"],
                        help="Set logging level (debug, info, no - warning, error, critical). ")
    args = parser.parse_args()

    infer_logger = logging.getLogger("inference.py")

    setup_logging(args.log, infer_logger)
    infer_logger.info("Logging is set to %s", args.log)

    # Load the model
    model = Transformer()
    
    # Load context and query
    context = torch.load(args.context)
    query = torch.load(args.query)

    # Tokenize context and query
    tokenizer = BytePairEncoder()   
    context_tokens = tokenizer.tokenize(context)
    query_tokens = tokenizer.tokenize(query)
    
    # Encode context and query
    encoded_context = model.encode(context_tokens)
    encoded_query = model.encode(query_tokens)
    
    # Perform inference (searching/sampling)
    results_tokens = model.infer(encoded_context, encoded_query)
    results = tokenizer.detokenize(results_tokens)

    print("Inference results:\n", results)

if __name__ == "__main__":
    main()