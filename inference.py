## 1. Load Context and Query
## 2. Tokenize Context and Query
## 3. Load Model
## 4. Encode Context and Query
## 5. Inference (Searching / Sampling)

import argparse
import logging
import torch
from models import Transformer, BytePairEncoder

def main():
    parser = argparse.ArgumentParser(description="Inference argument parser")
    parser.add_argument("--context", type=str, required=True, help="Context input file path")
    parser.add_argument("--query", type=str, required=True, help="Query input file path")

    # y : enable logging, n : disable logging
    parser.add_argument("--log", type=str, default="info", choices=["debug", "info", "no"],
                        help="Set logging level (debug, info, no - warning, error, critical). ")
    args = parser.parse_args()

    infer_logger = logging.getLogger("inference.py")
    infer_logger.setLevel(logging.INFO)  # Set to DEBUG for detailed logs

    if args.log == "debug":
        file_handler = logging.FileHandler()
        file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs 

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set to INFO for less verbose logs

        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        infer_logger.info("Logging is set to %s", args.log)

    elif args.log == "info":
        file_handler = logging.FileHandler()
        file_handler.setLevel(logging.INFO)  # Set to DEBUG for detailed logs 

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set to INFO for less verbose logs

        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        infer_logger.info("Logging is set to %s", args.log)

    else:
        file_handler = logging.FileHandler()
        file_handler.setLevel(logging.WARNING)  # Set to DEBUG for detailed logs 

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Set to INFO for less verbose logs

        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        infer_logger.info("Logging is set to %s", args.log)
    
    for logger in [Transformer.logger, BytePairEncoder.logger]:
        logger.setLevel(logging.DEBUG)
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

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