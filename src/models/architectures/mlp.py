"""
For this MLP implementation only node features x are considered.
The input array is a flattened array of length
num_static_node_features * num_timesteps_in.
"""

from torch import nn

from src.models.layers.mlp_layer import MLPLayer


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_timesteps_in = args.num_timesteps_in
        self.num_timesteps_out = args.num_timesteps_out
        self.num_static_node_features = args.num_static_node_features
        self.output_features = 1
        self.d_model = 512
        self.layers = 2
        self.dropout = 0.05
        self.activation = "gelu"

        # Encoder
        self.mlp = nn.ModuleList(
            [
                MLPLayer(
                    input_size=(
                        self.d_model
                        if i != 0
                        else (self.num_static_node_features * self.num_timesteps_in)
                    ),
                    output_size=self.d_model,
                    dropout=self.dropout,
                    activation=self.activation,
                    norm_layer="layer",
                )
                for i in range(self.layers)
            ]
        )
        self.projection = nn.Linear(
            self.d_model, (self.output_features * self.num_timesteps_out), bias=True
        )

    def forward(self, x, *_, **__):

        # Reshape input (flattening)
        # (n_windfarms, num_static_node_features, num_timesteps_in)
        # --> (n_windfarms, num_static_node_features * num_timesteps_in)
        outputs = x.reshape(x.shape[0], -1)

        # Pass through MLP
        for layer in self.mlp:
            outputs = layer(outputs)

        # Project
        outputs = self.projection(outputs)

        return outputs
