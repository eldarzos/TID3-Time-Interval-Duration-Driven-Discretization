from abc import ABC, abstractmethod
import pandas as pd


class TAMethod(ABC):
    """Abstract base class for a temporal (state) abstraction method."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Learn discretization states (i.e. bin boundaries) from training data."""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned states to data to generate a symbolic time series."""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)

    @abstractmethod
    def get_states(self):
        """Return the computed states (i.e. the bin boundaries)."""
        pass
