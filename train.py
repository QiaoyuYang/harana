# The entry point of the training code

from harana.train import main
from harana.utils import chord

import torch
import torch.nn as nn
import sys
torch.manual_seed(0)

if __name__ == "__main__":
    main(sys.argv[1:])

