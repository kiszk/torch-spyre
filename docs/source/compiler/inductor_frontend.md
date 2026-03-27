# Inductor Front-End: Deep Dive

This page provides a detailed reference for the Torch-Spyre Inductor
front-end compiler. For a high-level overview of the full compilation
pipeline, see [Compiler Architecture](architecture.md).

:::{figure} ../_static/images/torch-spyre-compilation-spectrum.png
:alt: Torch-Spyre compilation pipeline showing upstream versus custom components
:width: 95%
:align: center

The Torch-Spyre compilation pipeline. The left end (green) is entirely upstream PyTorch — Dynamo/Autograd and Inductor. The right end (pink) is Torch-Spyre's custom Inductor backend, which generates OpSpecs, SuperDSCs, and host code. Torch-Spyre also adds configurations and extensions to the upstream stages to tailor them for the Spyre device.
:::

## Extension Points

The front-end hooks into PyTorch Inductor via three extension points,
all registered in
[passes.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/passes.py):

| Extension Point | Stage | Purpose |
|----------------|-------|---------|
| `CustomPrePasses` | FX Graph (pre-lowering) | Graph rewrites before decomposition |
| `CustomPostPasses` | FX Graph (late post-grad) | Graph rewrites late in the post-grad FX graph passes |
| `_pre_fusion_custom_pass` | Scheduler | Kernel fusion and scheduling decisions (registered via `scheduler_passes` function) |

## Decompositions

Spyre-specific decompositions are registered with `@register_spyre_decomposition`
in
[decompositions.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/decompositions.py).
Decompositions transform complex ATen operations into simpler primitives
before the graph is lowered to loop-level IR.

## Lowerings

ATen operations are lowered to Inductor's LoopLevelIR in
[lowering.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/lowering.py)
using the `@register_spyre_lowering` decorator.

## Custom Operations

Spyre-specific operations with no ATen equivalent are defined in
[customops.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/customops.py)
using `@torch.library.custom_op`. Each custom op requires:

1. A signature definition (`@custom_op`)
2. A fake/meta function (`@opname.register_fake`)
3. Either a lowering + `SpyreOpFuncs` entry, or a decomposition that
   removes it from the graph before lowering

## Kernel Compilation: LoopLevelIR → OpSpec

[spyre_kernel.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/spyre_kernel.py)
compiles LoopLevelIR into `OpSpec` — a high-level, operation-level
description for the hardware. `SpyreOpFuncs` maps ATen operation names
to the corresponding Spyre OpFunc implementations.

## Code Generation: OpSpec → SuperDSC

The
[codegen/](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/codegen/)
package translates `OpSpec` into SuperDSC JSON — the input format
for the DeepTools back-end compiler.

## Stickification Pass

The stickification pass
([stickify.py](https://github.com/torch-spyre/torch-spyre/blob/main/torch_spyre/_inductor/stickify.py))
inserts layout conversion operations to ensure tensors are in the
correct Spyre memory layout (stick format) before each kernel.

## Adding a New Operation

See [Adding Operations](adding_operations.md) for a
step-by-step guide.
