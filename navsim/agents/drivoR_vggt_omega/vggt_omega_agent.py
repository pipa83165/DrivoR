from navsim.agents.drivoR.drivor_agent import DrivoRAgent

from .vggt_omega_features import VggtOmegaFeatureBuilder


class DrivoRVggtOmegaAgent(DrivoRAgent):
    """DrivoR agent using the VGGT-Omega feature builder and backbone."""

    def get_feature_builders(self):
        return [VggtOmegaFeatureBuilder(config=self._config)]
