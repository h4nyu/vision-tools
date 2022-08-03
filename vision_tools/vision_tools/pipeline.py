from __future__ import annotations

import pickle
from typing import Any, Callable, Optional


class CacheAction:
    def _persist(self, key: str, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            res = func(*args, **kwargs)
            self.save(key, res)

        return wrapper

    def save(self, key: str, obj: Any) -> None:
        with open(key, "wb") as fp:
            pickle.dump(obj, fp)

    def load(self, key: str) -> Any:

        with open(key, "rb") as fp:
            return pickle.load(fp)

    def __call__(
        self,
        fn: Any,
        args: list[Any] = [],
        kwargs: dict[str, Any] = {},
        kwargs_fn: Optional[Callable[None, dict]] = lambda: {},
        args_fn: Optional[Callable[None, list]] = lambda: [],
        key: str = None,
    ) -> tuple[Any, list, dict]:
        if key is not None:
            fn = self._persist(key, fn)

        def wrapper() -> Any:
            _kwargs = {}
            _args = []
            res = fn(*[*args, *args_fn()], **{**kwargs, **kwargs_fn()})

        return (wrapper, [], {})
