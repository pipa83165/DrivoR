from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.vggt_geometry import (
    VGGT_GEOMETRY_CAMERA_ORDER,
    preprocess_arrays_for_teacher,
)


class VggtOmegaFeatureBuilder(DrivoRFeatureBuilder):
    """Build unnormalized, officially resized VGGT-Omega camera inputs."""

    def get_unique_name(self) -> str:
        return "drivor_vggt_omega_feature"

    def _get_camera_feature(self, agent_input):
        cameras = agent_input.cameras[-1]
        raw_images = []
        for camera_name in VGGT_GEOMETRY_CAMERA_ORDER:
            image = getattr(cameras, camera_name).image
            if image is None:
                raise ValueError(f"Missing VGGT-Omega backbone camera image: {camera_name}")
            raw_images.append(image)
        images = preprocess_arrays_for_teacher(
            raw_images,
            mode=self._config.get("vggt_preprocess_mode", "balanced"),
            image_resolution=int(self._config.get("vggt_image_resolution", 512)),
        )
        return {"image": images}
