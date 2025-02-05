import torch

def save_model(model, filepath):
    """Save the model weights to the specified filepath."""
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    """Load the model weights from the specified filepath."""
    model.load_state_dict(torch.load(filepath))
    model.eval()