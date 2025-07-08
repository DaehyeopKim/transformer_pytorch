import torch
import logging

def logging_setup():
    """
    Set up logging configuration for the utils.
    """
    global logger
    logger = logging.getLogger("utils/utils.py")

logging_setup()


def load_transformer(model, file_path, device):
    """
    Load a transformer model from state dictionary.
    
    Args:
        model (torch.nn.Module): Model instance to load state into.
        file_path (str): Path to the state dictionary file.
        device (torch.device): Device to load the model onto.
        
    Returns:
        torch.nn.Module: Model with loaded state.
    """
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def save_transformer(model, file_path):
    """
    Save a transformer model's state dictionary to a file.
    
    Args:
        model (torch.nn.Module): Model to save.
        file_path (str): Path to save the state dictionary file.
    """
    torch.save(model.state_dict(), file_path)
    logger.info(f"Model saved to {file_path}")

def save_checkpoint(model, optimizer, epoch, file_path, loss):
    """
    Save a model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Current epoch number.
        file_path (str): Path to save the checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss' : loss
    }
    torch.save(checkpoint, file_path)
    logger.info(f"Checkpoint saved to {file_path}")

def load_checkpoint(model, optimizer, file_path, device):
    """
    Load a model checkpoint.
    
    Args:
        model (torch.nn.Module): Model instance to load state into.
        optimizer (torch.optim.Optimizer): Optimizer instance to load state into.
        file_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.to(device)
    logger.info(f"Checkpoint loaded from {file_path}")
    logger.info(f"Resumed from epoch {epoch} with loss {loss}")
    
    return model, optimizer, epoch, loss

def save_transformer_with_config(model, file_path):
    """
    Save a transformer model with its configuration.
    
    Args:
        model (torch.nn.Module): Transformer model to save.
        file_path (str): Path to save the model file.
    """
    # Use the stored config from the model if available
    if hasattr(model, 'config'):
        model_config = model.config.copy()
    else:
        # Fallback to extracting config from model structure (for backward compatibility)
        model_config = {
            'token_size': model.token_size,
            'embed_dim': model.embedding.embedding.embedding_dim,
            'ffn_dim': model.encoder.blocks[0].feed_forward.w1.out_features,
            'head_num': model.encoder.blocks[0].self_attention.head_num,
            'k_dim': model.encoder.blocks[0].self_attention.k_dim,
            'v_dim': model.encoder.blocks[0].self_attention.v_dim,
            'N': len(model.encoder.blocks),
            'eos_token_id': model.eos_token_id,
            'bos_token_id': model.bos_token_id,
            'model_name': model.model_name,
            'tokenizer_name': getattr(model, 'tokenizer_name', 'BytePairEncoder')
        }
    
    # Save both state dict and config
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }
    
    torch.save(save_data, file_path)
    logger.info(f"Model with configuration saved to {file_path}")

def load_transformer_with_config(file_path, device):
    """
    Load a transformer model with its configuration.
    
    Args:
        file_path (str): Path to the model file.
        device (torch.device): Device to load the model onto.
        
    Returns:
        torch.nn.Module: Loaded transformer model.
    """
    # Load the saved data
    save_data = torch.load(file_path, map_location=device)
    model_config = save_data['model_config']
    
    # Import Transformer class (assuming it's available)
    from models.transformer import Transformer
    
    # Create model instance with saved configuration
    # Use ** to unpack the config dictionary
    model = Transformer(**model_config)
    
    # Load the state dict
    model.load_state_dict(save_data['model_state_dict'])
    model.to(device)
    
    logger.info(f"Model loaded from {file_path}")
    logger.info(f"Model config: {model_config}")
    
    return model

def save_checkpoint_with_config(model, optimizer, epoch, file_path, loss):
    """
    Save a model checkpoint with configuration.
    
    Args:
        model (torch.nn.Module): Transformer model to save.
        optimizer (torch.optim.Optimizer): Optimizer state to save.
        epoch (int): Current epoch number.
        file_path (str): Path to save the checkpoint file.
        loss (float): Training loss value.
    """
    # Use the stored config from the model if available
    if hasattr(model, 'config'):
        model_config = model.config.copy()
    else:
        # Fallback to extracting config from model structure (for backward compatibility)
        model_config = {
            'token_size': model.token_size,
            'embed_dim': model.embedding.embedding.embedding_dim,
            'ffn_dim': model.encoder.blocks[0].feed_forward.w1.out_features,
            'head_num': model.encoder.blocks[0].self_attention.head_num,
            'k_dim': model.encoder.blocks[0].self_attention.k_dim,
            'v_dim': model.encoder.blocks[0].self_attention.v_dim,
            'N': len(model.encoder.blocks),
            'eos_token_id': model.eos_token_id,
            'bos_token_id': model.bos_token_id,
            'model_name': model.model_name,
            'tokenizer_name': getattr(model, 'tokenizer_name', 'BytePairEncoder')
        }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'loss': loss
    }
    
    torch.save(checkpoint, file_path)
    logger.info(f"Checkpoint with configuration saved to {file_path}")

def load_checkpoint_with_config(file_path, device, optimizer_class=None, optimizer_kwargs=None):
    """
    Load a model checkpoint with configuration.
    
    Args:
        file_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        optimizer_class: Optimizer class (e.g., torch.optim.Adam). If None, optimizer won't be created.
        optimizer_kwargs (dict): Optimizer initialization arguments.
        
    Returns:
        tuple: (model, optimizer, epoch, loss) or (model, None, epoch, loss) if optimizer_class is None
    """
    # Load checkpoint
    checkpoint = torch.load(file_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Import Transformer class
    from models.transformer import Transformer
    
    # Create model instance with saved configuration
    # Use ** to unpack the config dictionary
    model = Transformer(**model_config)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create and load optimizer if specified
    optimizer = None
    if optimizer_class is not None:
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {file_path}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Resumed from epoch {epoch} with loss {loss}")
    
    return model, optimizer, epoch, loss