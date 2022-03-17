from __future__ import annotations

import pickle
from typing import Any, Callable


class CacheAction:
    def _persist(self, key: str, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            res = func(*args, **kwargs)
            with open(key, "wb") as fp:
                pickle.dump(res, fp)

        return wrapper

    def __call__(
        self,
        fn: Any,
        args: list[Any] = [],
        kwargs: dict[str, Any] = {},
        output_args: list[str] = [],
        output_kwargs: dict[str, str] = {},
        key: str = None,
    ) -> tuple[Any, list, dict]:
        if key is not None:
            fn = self._persist(key, fn)

        def wrapper() -> Any:
            _kwargs = {}
            _args = []
            for k, v in output_kwargs.items():
                with open(v, "rb") as fp:
                    _kwargs[k] = pickle.load(fp)
            for v in output_args:
                with open(v, "rb") as fp:
                    _args.append(pickle.load(fp))
            res = fn(*[*args, *_args], **{**kwargs, **_kwargs})

        return (wrapper, [], {})
