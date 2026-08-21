from abc import ABC, abstractmethod
from typing import Any, Dict


class PhysicsEngine(ABC):
    """
    Abstract physics engine wrapper, decoupling raw interaction (e.g., PyBullet, MuJoCo)
    from the simulator generation process.
    """

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, dt: float) -> None:
        pass

    @abstractmethod
    def add_body(self, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def remove_body(self, body_id: str) -> None:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass


class DummyPhysicsEngine(PhysicsEngine):
    """
    A minimal, mock physics engine to prove the structure works before wiring up MuJoCo/PyBullet.
    """

    def __init__(self):
        self.bodies = {}
        self.time = 0.0

    def reset(self) -> None:
        self.bodies.clear()
        self.time = 0.0

    def step(self, dt: float) -> None:
        self.time += dt

    def add_body(self, **kwargs: Any) -> str:
        body_id = f"body_{len(self.bodies)}"
        self.bodies[body_id] = kwargs
        return body_id

    def remove_body(self, body_id: str) -> None:
        if body_id in self.bodies:
            del self.bodies[body_id]

    def get_state(self) -> Dict[str, Any]:
        return {"time": self.time, "bodies": self.bodies.copy()}
