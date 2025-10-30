import inspect
import functools
from typing import Annotated, get_origin, get_args

from .context import Context, FieldUpdates
from dataclasses import is_dataclass


def _get_value(ctx: Context, name, annotation: any):
    call_args = {}
    ctx_var = ctx.fields

    is_annotated = (get_origin(annotation) is Annotated)
    var_name = "fields"
    if is_annotated:
        var_name = get_args(annotation)[1]
        ctx_var = getattr(ctx, var_name)

    if name in ctx_var:
        call_args[name] = ctx_var[name]
    else:
        raise KeyError(
            f"Required parameter '{name}' "
            f"was not found in {var_name} and has no default value."
        )
    
    return call_args

def step(wrapped_func: callable) -> callable:
    """
    Decorator that modifies a function to be called with a Context object.
    It inspects the function's signature and retrieves the
    required arguments from the context.fields dictionary.
    """
    # Get the function's signature and parameter names ONCE
    sig = inspect.signature(wrapped_func)
    param_names = sig.parameters

    @functools.wraps(wrapped_func)
    def context_wrapper(context: Context):
        # Build the arguments for the function call
        call_args = {}
        
        for name in param_names:
            param = sig.parameters[name]
            annotation = param.annotation
            if is_dataclass(annotation):
                dc_sig = inspect.signature(annotation)

                dc_args = {}

                for dc_name, dc_param in dc_sig.parameters.items():
                    dc_annotation = dc_sig.parameters[dc_name].annotation
                    dc_args.update(_get_value(context,dc_name,dc_annotation))
                    
                call_args[name] = annotation(**dc_args)
            else:
                call_args.update(_get_value(context,name,annotation))
           

        # Call the original function with the unpacked arguments
        print(f"--- Running step: {wrapped_func.__name__} ---")
        results = wrapped_func(**call_args) 

        # 3. Process the results
        
        # Normalize results to a tuple to handle single or multiple returns
        if not isinstance(results, tuple):
            results = (results,) # Make it a single-item tuple

        for item in results:
            if isinstance(item, FieldUpdates):
                print(f"--- Updating fields with: {item} ---")
                context.fields.update(item)            
            else:
                print(f"--- Warning: Step '{wrapped_func.__name__}' returned an unprocessed object: {type(item)} ---")
    
    wrapped_func.run = context_wrapper

    return wrapped_func
