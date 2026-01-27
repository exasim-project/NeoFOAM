# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Init class for Architecture 5b - FastAPI-style instance with decorators.

Usage (like FastAPI):
    init = Init("SimpleSolver")

    @init.step
    def runTime() -> pyf.Time:
        return pyf.Time.read(argv)

    @init.step
    def mesh(runTime: Annotated[pyf.Time, Depends(runTime)]) -> pyf.fvMesh:
        return pyf.fvMesh.read(runTime)
"""

from typing import (
    Any,
    Callable,
    TypeVar,
    Annotated,
    get_type_hints,
    get_origin,
    get_args,
)
from functools import wraps
import inspect

from foamadapter.framework.context import Context
from .depends import Depends


T = TypeVar("T")


class Init:
    """
    FastAPI-style Init instance with decorator methods.

    Usage (like FastAPI app):
        init = Init("SimpleSolver")

        @init.step
        def runTime() -> pyf.Time:
            return pyf.Time.read(argv)

        @init.step
        def mesh(runTime: Annotated[pyf.Time, Depends(runTime)]) -> pyf.fvMesh:
            return pyf.fvMesh.read(runTime)

        @init.step
        def field_p(mesh: Annotated[pyf.fvMesh, Depends(mesh)]) -> pyf.volScalarField:
            return pyf.volScalarField.read(mesh, "p")

    Compare to FastAPI:
        app = FastAPI()

        @app.get("/")
        def root():
            return {"message": "Hello"}
    """

    def __init__(self, name: str, argv: list[str] | None = None):
        """
        Args:
            name: Name of the solver/model this init belongs to
            argv: Command-line arguments (for OpenFOAM initialization)
        """
        self.name = name
        self.argv = argv or []
        self._steps: dict[str, Callable] = {}
        self._cache: dict[str, Any] = {}
        self._build_context_func: Callable | None = None

    def step(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to register an init step (like @app.get in FastAPI).

        Usage:
            @init.step
            def mesh(runTime: Annotated[pyf.Time, Depends(runTime)]) -> pyf.fvMesh:
                return pyf.fvMesh.read(runTime)

        The decorator:
        - Registers the function as an init step
        - Wraps it with caching and dependency resolution
        - Returns the wrapped function (can still be called directly)
        """

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Check cache first
            cache_key = func.__name__
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Resolve dependencies from Annotated type hints
            resolved_args = self._resolve_dependencies(func)

            # Merge with provided kwargs
            resolved_args.update(kwargs)

            # Execute with resolved dependencies
            result = func(*args, **resolved_args)

            # Cache result
            self._cache[cache_key] = result
            return result

        wrapper._is_init_step = True
        wrapper._original_func = func
        self._steps[func.__name__] = wrapper
        return wrapper

    def build_context(self, func: Callable[..., Context]) -> Callable[..., Context]:
        """
        Decorator to register the build_context function.

        Usage:
            @init.build_context
            def build() -> Context:
                ctx = Context()
                ctx.runTime = runTime()
                ctx.mesh = mesh()
                return ctx
        """
        self._build_context_func = func
        return func

    def _resolve_dependencies(self, func: Callable) -> dict[str, Any]:
        """
        Resolve dependencies from Annotated type hints (FastAPI-style).

        Looks for parameters with Annotated[Type, Depends(other_func)] and
        calls the dependency function to get the value.
        """
        resolved = {}

        try:
            hints = get_type_hints(func, include_extras=True)
        except Exception:
            return resolved

        sig = inspect.signature(func)

        for param_name, param in sig.parameters.items():
            hint = hints.get(param_name)
            if hint is None:
                continue

            # Check if it's Annotated[Type, Depends(...)]
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                for arg in args[1:]:  # Skip first arg (the actual type)
                    if isinstance(arg, Depends):
                        # Call the dependency function
                        resolved[param_name] = arg()
                        break

        return resolved

    def run(self) -> Context:
        """
        Execute build_context and return the Context.

        This is called by the Solver to get the initialized Context.
        """
        if self._build_context_func is None:
            raise RuntimeError(f"No @{self.name}.build_context defined")
        return self._build_context_func()

    def clear_cache(self) -> None:
        """Clear the cached values (useful for testing)."""
        self._cache.clear()

    def __getitem__(self, name: str) -> Callable:
        """Get a registered step by name."""
        return self._steps[name]


class ModelInit(Init):
    """
    Init for optional models - extends existing Context.

    Usage:
        model_init = ModelInit("boussinesq")

        @model_init.step
        def field_T(mesh: Annotated[Mesh, Depends(get_mesh)]) -> Field:
            return Field.read(mesh, "T")

        @model_init.extend_context
        def extend(ctx: Context) -> Context:
            ctx.fields["T"] = field_T()
            return ctx
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._extend_context_func: Callable | None = None

    def extend_context(
        self, func: Callable[[Context], Context]
    ) -> Callable[[Context], Context]:
        """
        Decorator to register the extend_context function.

        Usage:
            @model_init.extend_context
            def extend(ctx: Context) -> Context:
                ctx.fields["T"] = field_T()
                return ctx
        """
        self._extend_context_func = func
        return func

    def extend(self, ctx: Context) -> Context:
        """Execute extend_context on an existing Context."""
        if self._extend_context_func is None:
            raise RuntimeError(f"No @{self.name}.extend_context defined")
        return self._extend_context_func(ctx)
