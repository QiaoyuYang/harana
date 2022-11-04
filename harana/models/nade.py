# Adapt from 
# https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/autoregressive/nade.py
# and https://gitlab.com/algomus.fr/functional-harmony/-/blob/master/frog/models/nade_tools.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockNADE(nn.Module):
    """The Neural Autoregressive Distribution Estimator (NADE) model."""

    def __init__(self, embedding_size, visible_dim_list, hidden_dim):
        """Initializes a new NADE instance.
        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimmension of the hidden layer. NADE only supports one
                hidden layer.
        """
        super(BlockNADE, self).__init__()

        self.visible_dim_list = visible_dim_list
        self.hidden_dim = hidden_dim
        self.num_visible_units = len(self.visible_dim_list)
        # fmt: off
        self.W_weight = nn.ParameterList([torch.zeros(self.hidden_dim, visible_dim) for visible_dim in self.visible_dim_list])
        #self.W_bias = nn.Parameter(torch.zeros(hidden_dim,))
        self.V_weight = nn.ParameterList([torch.zeros(visible_dim, self.hidden_dim) for visible_dim in self.visible_dim_list])
        #self.V_bias = nn.Parameter(torch.zeros(self.input_dim,))

        self.linear2W_bias = nn.Linear(embedding_size, self.hidden_dim)
        self.linear2V_bias = nn.ModuleList([nn.Linear(embedding_size, visible_dim) for visible_dim in self.visible_dim_list])

    def _forward(self, x):
        """Computes the forward pass and samples a new output.
        Returns:
            (p_hat, x_hat) where p_hat is the probability distribution over dimensions
            and x_hat is sampled from p_hat.
        """

        W_bias = torch.sigmoid(self.linear2W_bias(x))
        V_bias = [torch.sigmoid(l2vb(x)) for l2vb in self.linear2V_bias]

        '''
        # If the input is an image, flatten it during the forward pass.
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(original_shape[0], -1)

        p_hat, x_hat = [], []
        batch_size = 1 if x is None else x.shape[0]
        # Only the bias is used to compute the first hidden unit so we must replicate it
        # to account for the batch size.
        a = self.W_bias.expand(batch_size, -1)
        for i in range(self.input_dim):
            h = torch.relu(a)
            p_i = torch.sigmoid(h @ self.V_weight[i : i + 1, :].t() + self.V_bias[i : i + 1])
            p_hat.append(p_i)

            # Sample 'x' at dimension 'i' if it is not given.
            x_i = x[:, i : i + 1]
            x_i = torch.where(x_i < 0, distributions.Bernoulli(probs=p_i).sample(), x_i)
            x_hat.append(x_i)

            # We do not need to add self.in_b[i:i+1] when computing the other hidden
            # units since it was already added when computing the first hidden unit.
            a = a + x_i @ self.W_weight[:, i : i + 1].t()
        if x_hat:
            return (
                torch.cat(p_hat, dim=1).view(original_shape),
                torch.cat(x_hat, dim=1).view(original_shape),
            )
        '''
        return []
    def forward(self, x):
        """Computes the forward pass.
        Args:
            x: Either a tensor of vectors with shape (n, input_dim) or images with shape
                (n, 1, h, w) where h * w = input_dim.
        Returns:
            The result of the forward pass.
        """
        self._forward(x)
        return 0

    def sample(self, n_samples=None, conditioned_on=None):
        """See the base class."""
        with torch.no_grad():
            conditioned_on = self._get_conditioned_on(n_samples, conditioned_on)
            return self._forward(conditioned_on)[1]


