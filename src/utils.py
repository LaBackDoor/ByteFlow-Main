from src2.model.config import ByT5Config
from src2.model.model import ByT5ForConditionalGeneration
import torch.nn as nn

# Create and initialize a ByT5-Small model
def create_byt5_small():
    config = ByT5Config()
    model = ByT5ForConditionalGeneration(config)

    # Initialize weights (simplified, production code would need better initialization)
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    model.apply(_init_weights)

    return model