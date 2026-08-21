from abc import ABC, abstractmethod
from typing import Any, Dict


class GeometryAPI(ABC):
    """
    Decoupled vision/geometry API replacing the monolithic API base in landgen.
    Provides methods solely for scene understanding, point clouds, meshing.
    """

    @abstractmethod
    def extract_point_cloud(self, image: Any) -> Any:
        pass

    @abstractmethod
    def fit_primitive(self, point_cloud: Any) -> Dict[str, Any]:
        pass


class DummyGeometryAPI(GeometryAPI):
    """
    Mock implementation of Geometry for Sprint 2.
    """

    def extract_point_cloud(self, image: Any) -> Any:
        return "mock_point_cloud"

    def fit_primitive(self, point_cloud: Any) -> Dict[str, Any]:
        return {"shape": "box", "size": [1.0, 1.0, 1.0]}
