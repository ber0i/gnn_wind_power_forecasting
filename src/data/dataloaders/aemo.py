import numpy as np
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class DataLoader:
    """
    Dataset loader for the AEMO wind farm data.
    """

    def __init__(self, args):
        super().__init__()
        self.features = []
        self.targets = []
        self.correlation_treshold = args.correlation_threshold
        self.use_wind_direction = args.use_wind_direction
        self.use_wind_speed = args.use_wind_speed
        self._read_data()

    def _read_data(self):

        edge_index = np.load(
            f"data/processed/aemo/2022/edge_index_{self.correlation_treshold}.npy",
        )
        edge_attr = np.load(
            f"data/processed/aemo/2022/edge_attr_{self.correlation_treshold}.npy",
        )
        X = np.load("data/processed/aemo/2022/x.npy")

        # select the features to use for modeling
        feature_selection_indices = [0]
        if self.use_wind_direction:
            feature_selection_indices.append(1)
        if self.use_wind_speed:
            feature_selection_indices.append(2)

        X = X[:, feature_selection_indices, :]

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Scale edge weights between 0 and 1
        edge_attr = (edge_attr - np.min(edge_attr)) / (
            np.max(edge_attr) - np.min(edge_attr)
        )
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Normalise X with z-score method
        # Compute mean over dimension 0 (wind farms) and 2 (time)
        # This results in one mean value per feature
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)  # reshape transforms to shape (1, n_feature, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        self.X = torch.tensor(X, dtype=torch.float)

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """
        Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average wind speed using num_timesteps_in to predict the
        wind speed in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for AEMO dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The AEMO wind power
                forecasting dataset.
        """
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edge_index, self.edge_attr, self.features, self.targets
        )

        return dataset
