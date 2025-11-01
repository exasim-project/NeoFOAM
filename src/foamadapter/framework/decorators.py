



# from dataclasses import dataclass, field


# @dataclass
# class StepMeta:
#     name: str
#     order: int
#     repeat: str
#     func: callable
#     depends_on: list = field(default_factory=list)

# def step(order, repeat=None, depends_on=None):
#     def decorator(func):
#         func._step_meta = StepMeta(
#             name=func.__name__,
#             order=order,
#             repeat=repeat,
#             func=func,
#             depends_on=depends_on or [],
#         )
#         return func
#     return decorator

# def model(name):
#     def decorator(cls):
#         # register_model(name, cls)
#         cls.__model_name__ = name
#         # Collect step metadata
#         steps = []
#         for attr in dir(cls):
#             fn = getattr(cls, attr)
#             meta = getattr(fn, "_step_meta", None)
#             if meta:
#                 steps.append(meta)
#         cls._steps = steps
#         @classmethod
#         def count_steps(cls_):
#             return len(cls_._steps)
#         cls.count_steps = count_steps
#         return cls
#     return decorator
