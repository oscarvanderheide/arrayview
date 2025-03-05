from typing import Callable, Dict, List
from dataclasses import dataclass, field

@dataclass
class ArrayShowEventSystem:
    subscribers: Dict[str, List[Callable]] = field(default_factory=lambda: {
        'state_changed': [],
        'scroll_changed': [],
        'view_dims_changed': [],
        'dimension_text_changed': [],  # New event type
        'scroll_dim_changed': []  # New event type
    })

    def subscribe(self, event_type: str, callback: Callable) -> None:
        if event_type in self.subscribers:
            self.subscribers[event_type].append(callback)

    def emit(self, event_type: str, *args, **kwargs) -> None:
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(*args, **kwargs)