import torch
import logging


def logging_setup():
    """
    Set up logging configuration for the transformer model.
    This function is executed when "--log" option is called in "inference.py", "train.py", "test.py".
    """
    global logger
    logger = logging.getLogger("models/transformer.py")

    file_handler = logging.FileHandler()
    file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs 

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set to INFO for less verbose logs


    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
logging_setup()

def generate_mask(seq_len, device=None):
    """
    Generate a square mask for the sequence length.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask

class FeedForward:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2

class Attention:
    def __init__(self, model_name, self_attention=False):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class SingleHeadAttention:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2

class MultiHeadAttention:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class DecoderBlock:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
    
class EncoderBlock:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2

class Transformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def forward(self, x):
        # Dummy forward method
        return x * 2
