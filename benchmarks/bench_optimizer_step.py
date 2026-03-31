import cProfile
import pstats
from enum import StrEnum
from math import prod
from time import perf_counter
import numpy as np
import torch
import typer

import heavyball

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_SHAPES = ((128, 128),) * 4 + ((512, 128),) * 2 + ((512,),) * 2 + ((128,),) * 4


class DType(StrEnum):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"


def parse_shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(map(int, text.lower().replace("x", " ").split()))
    except ValueError as e:
        raise typer.BadParameter(f"invalid shape: {text!r}") from e
    if not shape:
        raise typer.BadParameter(f"invalid shape: {text!r}")
    return shape


@app.command()
def main(optimizer: str = "AdamW", dtype: DType = DType.float32, shape: list[str] | None = None,
        compile_step: bool = False, update_precond: bool | None = None, steps: int = 300, warmup: int = 20,
        windows: int = 6, seed: int = 0, ):
    shapes = DEFAULT_SHAPES if shape is None else tuple(map(parse_shape, shape))
    torch_dtype = getattr(torch, dtype)
    kwargs = {"compile_step": compile_step}
    if update_precond is not None:
        kwargs["preconditioner_update_probability"] = float(update_precond)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    params = []
    for dims in shapes:
        param = torch.nn.Parameter(torch.randn(dims, device="cuda", dtype=torch_dtype, generator=gen))
        param.grad = torch.randn(dims, device="cuda", dtype=torch_dtype, generator=gen)
        params.append(param)

    step = getattr(heavyball, optimizer)(params, **kwargs).step
    for _ in range(warmup):
        step()

    times = []
    for _ in range(windows):
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        times.append((perf_counter() - start) / steps)

    print(f"{len(shapes)} tensors, {sum(prod(s) for s in shapes)} total params")
    print(f'Median Time: {np.median(times) * 1e6:.3f}µs')

if __name__ == "__main__":
    app()
