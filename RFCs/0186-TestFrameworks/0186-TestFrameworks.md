
# RFC-0186: YAML-driven Torch.ops Test Framework for Spyre/AIU-backed Models

**Authors:**
- Tuan M. Hoang Trong (YKT)
- Kazuaki Ishizaki (TRL)
- Umamaheswari Devi (IRL)

**Tracking issue:** #241
**Target repository:** `torch-spyre/torch-spyre`
**Related issues/PRs:** #323

---

## Summary

We propose a YAML-driven test framework (built on `pytest`) to validate functional correctness of `torch.ops` used by Spyre/AIU-supported models.
Outside the scope of this RFC, there is an existing internal framework to extract per-model sets of `torch.ops` and their inputs via a scanning codegen pass. This internal framework is modified accordingly to emits compact YAML descriptors instead of per-op Python test files.
This RFC does not propose changes to the scanning logic beyond YAML emission.

The proposed framework:

- Accept the generated YAML files
- Generates and runs tests from YAML, with optional deduplication across models.
- Pytest markers & selection: We use pytest.mark (registered in pytest.ini) for flexible selection, skipping, expected failures, etc.
- Integrates with CI as nightly tests (not every PR), initially focusing on `*-spyre.yaml` variants, then converging to canonical `<model>.yaml` once torch-spyre stabilizes.

This design reduces boilerplate, centralizes test intent, simplifies maintenance, and lets us scale coverage across models with consistent correctness checks.

**Amenability to broader use-cases**:
While this RFC focuses on validating `torch.ops` for Spyre/AIU-backed models, the proposed YAML-driven framework is generic by design. It can be applied to other scenarios where:

- Hand-written YAML manifests are preferred for custom operator testing.
- Backend-specific correctness checks are needed outside Spyre/AIU.
- Debugging workflows require reproducible inputs from .pt files or controlled presets.

This flexibility makes the framework suitable for future extensions beyond the scope of this RFC.

---

## Motivation

### Backgroud
In past, we met problems that the inference results are almost functinally correct but some of torch operations return incorrect results. This behavior is fragile and became obstacles to adding new features such as fine-tuning.
So, we believe that test cases per operation with actual parameters in models are crucial.

### Problem
Current test generation emits one Python file per op+input, leading to:
- Significant code duplication.
- Large and noisy commits that are hard to review and maintain.
- Difficulty updating shapes/inputs as models evolve.

### Goals
- **Functional correctness:** Validate each `torch.op` against a CPU reference for corresponding shapes and data.
- **Nightly monitoring:** Continuously track correctness of enabled `opFuncs`.
- **Maintainability:** Make test intent readable and updatable (shapes, dtypes, tolerances) in YAML.
- **Scalability:** Support multiple models (Granite-4h, GPT-oss, Granite-vision-3, Granite-speech) with cross-model deduplication.

### Non-goals (for now)
- **Performance tracking:** Only scoped as “N/A” initially (we list future scenarios and metrics below).
- **Per-PR gating:** Runs in nightly CI/CD, not on every PR.

---

## Proposed Implementation

### Overview
We use two cooperating frameworks:

1) **Model scan & codegen (existing, internal, modified):**  
   - Scan target models and extract the set of `torch.ops` along with input shapes and/or sample inputs.  
   - **Change:** Generate **YAML** instead of Python test files.

2) **YAML test runner (the proposed one):**  
   - Load YAML case definitions.  
   - Codegen lightweight test functions dynamically (via `pytest`).  
   - Execute ops, compare against CPU reference, and report results.
   - Skip tests if needed.

---
### Execution strategy

- Run all extracted ops for a model (subject to dedupe).
- Avoid redundant runs (global cache keyed by <op, normalized_inputs>).
- Make test parameters easy to understand for debugging (schema + description fields).

---

### Models & files
We plan per-model YAML, with option for subclasses variants:

- `tests/_inductor/models/template.yaml`
- `tests/_inductor/models/gpt-oss.yaml`
- `tests/_inductor/models/gpt-oss-spyre.yaml`  
  *Temporary, backend-specific functional correctness checks. May relax original shapes and can be deleted once features are stable.*

**Naming:** (notice to changes in internal scripts) Permanent test cases (e.g., no `_fp16`) reside in files **without** postfixes. Backend-temporary cases use the `-spyre` postfix file.

---

### Deduplication & selection
- **Cross-model deduplication (default):** If `op-A(input-B)` was already tested for model-C, skip rerunning for model-D, if that test case is part of model-D.
Deduplication is based on normalized op signature and input shapes, not model name.
- **Disable dedupe:** `--no-dedupe` (explicit).
- **Selective model runs:** Choose one or many models at runtime (e.g., run only `granite3-speech`).
- **Select cases**: integrate with pytest.ini design, register custom markers for cleanliness.
Marker usage and registration follow pytest documentation best practices, enabling selection and skip/xfail logic

**CLI Examples**

```bash
# List available models/cases
pytest --list-models 2>&1 | tee logs_test.txt
pytest --list-cases  2>&1 | tee logs_test.txt

# Run a single case by name
pytest -q tests/_inductor/test_model_ops.py --model gpt_oss -k "mul_ok"

# Run all 'torch.add' ops for GPT-oss
pytest -q tests/_inductor/test_model_ops.py --model gpt_oss -k "torch.add"

# Run for multiple models
pytest --model gpt_oss tests/
pytest --model gpt_oss --model granite4h tests/
```

**Full argument list**

supported arguments: pytest

```bash
pytest \
  [--model <model-name>]*
  --dedupe / --no-dedupe
  --list-models, --list-cases
  --compile-backend "inductor"
    TEST_COMPILE_BACKEND: env. variable to change the default backend “inductor”
```

---

### Source Layout

Below is the proposed directory structure for the YAML-driven test framework:

```bash
tests/
├── conftest.py
└── _inductor/
    ├── init.py
    ├── test_model_ops.py          # Entry point for pytest-based tests
    ├── model_cases_loader.py      # Loads and parses YAML files
    ├── op_registry.py             # Maintains mapping of ops to test logic
    ├── runner.py                  # Executes tests dynamically from YAML
    └── models/                    # YAML manifests per model
        ├── gpt_oss.yaml
        └── granite3_speech.yaml
```

## YAML Schema

Top-level structure

```yaml

model: <model-name>
default:
  dtype: <DTYPE_MAP value>
  seed: <int>
  rtol: <float>
  atol: <float>
cases:
  - name: <case-name>
    op: <torch.op>            # as registered via OP_REGISTRY
    inputs:                   # map of arguments for the op
      tensor:                 # tensor description (see below)
      value:                  # scalar input
      py:                     # restricted Python literal (tuples, None, Ellipsis, slice, ints, floats, lists, inf, -inf, nan)
      tensor_list:            # sequences (list/tuple) of tensors
      file:                   # .pt file holding values for a tensor
    description: <text>       # e.g., trace where the op was extracted
    expects_error:            # expected error (optional) - expected error type and message substring
      type: <ExceptionType>
      match: <substring>

    # Test control
    mark: <pytest marker(s)>
    skip_if:                  # Conditional skip logic for backend/device)
      expr: <python boolean>  # if True -> skip
      reason: <text>

    # Per-case overrides of defaults
    dtype: <DTYPE_MAP value>
    seed: <int>
    rtol: <float>
    atol: <float>

```

---

### Input to a test case

A test case is generated based on the op to test, and the input sources which is a sequence of arguments, each can be one of the following

- -tensor: randomly generated tensor(s) following shape, dtype, init, etc.
- -value: scalars, lists, tuples.
- -py: restricted Python literal expressions (tuples, None, Ellipsis, slices, ints, floats, lists, inf, -inf, nan).
- -file: load from .pt files (torch.load("path/to/file.pt")).

Rationale: Debugging error-sensitive bugs often requires specific tensor values, especially in the backward path, so YAML supports exact data (init: data) and file-based inputs (-file).

### Input tensor description

```yaml

tensor:
  shape: [dim0, dim1, ...]
  init: rand | zeros | ones | arange | randint | uniform | data   # optional
  dtype: <optional dtype>
  preset: <optional preset for non-contiguous/layout-sensitive ops>
  preset_args: <optional args for preset>

  # init_args specify bounds for random generation
  # If 'randint': allow 'low' (default 0), 'high'
  # If 'uniform': allow 'low' (default 0), 'high' (default 1)
  # If 'data': pass exact values as list(s)
```

---

### Supporting Constants

The framework uses predefined mappings for dtypes and error names to ensure consistency between YAML descriptors and Python runtime.

#### DTYPE_MAP
Maps string keys in YAML to actual `torch` dtypes:

```python
DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
}
```

##### Error type

The value in `expected_errors` is a string which is mapped to Python exception classes based on the following map:

```python
_ERR_NAME_TO_TYPE = {
    "RuntimeError": RuntimeError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "AssertionError": AssertionError,
}
```

---

### Example YAML File

Below is a sample `template.yaml` illustrating the proposed schema and advanced features such as dtype overrides, skip conditions, error expectations, and presets for non-contiguous layouts.

```yaml
model: template
default:
  dtype: fp16        # Default dtype for all cases unless overridden
  seed: 123          # Seed for reproducibility
  rtol: 1.0e-3       # Default relative tolerance
  atol: 1.0e-3       # Default absolute tolerance

cases:
  - name: add_basic
    op: torch.add
    inputs:
      - tensor: {shape: [4, 128, 768], init: randn}
      - tensor: {shape: [4, 128, 768], init: randn}
    description: |
      Example from GPT-oss forward path:
      File: site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:138
      Code: next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]

  - name: op_bool_tensors
    op: torch.logical_and
    inputs:
      - tensor: {shape: [11, 6], dtype: bool}
      - tensor: {shape: [11, 6], dtype: bool}
    description: |
      Ops with randomly generated bool tensors (50% chance of True)

  - name: op_different_dtypes
    op: torch.add
    inputs:
      - tensor: {shape: [11, 6], dtype: fp32, init: randn}
      - tensor: {shape: [11, 6], dtype: int64, init: randint, init_args: {high: 100}}
    description: |
      Ops with args of different dtypes

  - name: mul_tensor_scalar
    op: torch.mul
    inputs:
      - tensor: {shape: [1, 64], init: uniform, init_args: {low: -1.0, high: 1.0}}
      - value: 3.0
    description: |
      Ops with tensor and scalar as args

  - name: mul_inttensor_scalar
    op: torch.mul
    inputs:
      - tensor: {shape: [1, 64], init: randint, init_args: {low: 1, high: 100}}
      - value: 3
    description: |
      Ops with tensor and scalar as args

  - name: aten_view
    op: torch.ops.aten.view
    inputs:
      - tensor: {shape: [1, 64]}
      - value: [2, 32]
    description: |
      Value can accept scalar, list or tuple

  - name: softmax_lastdim
    op: torch.nn.functional.softmax
    attrs: {dim: -1}
    inputs:
      - tensor: {shape: [2, 4, 8], init: arange}

  - name: cat_list
    op: torch.cat
    attrs: {dim: -1}
    inputs:
      - tensor_list:
          - {shape: [2, 3, 4], init: randn}
          - {shape: [2, 3, 5], init: randn}

  - name: view_should_error_on_noncontig
    op: torch.view
    expects_error: {type: RuntimeError, match: view}
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: noncontig_slice
          preset_args: {dim: 1, step: 2}
      - value: [2, 24]

  - name: view_should_error_on_noncontig_167
    op: torch.view
    expects_error: {type: RuntimeError, match: view}
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: noncontig_slice
          preset_args: {dim: 1, step: 2}
      - value: 32
      - value: -1
      - value: 2880

  - name: contiguous_from_transpose_view
    op: torch.contiguous
    inputs:
      - tensor:
          shape: [2, 3, 8]
          preset: transpose_view
          preset_args: {dim0: 1, dim1: 2}

  - name: rsqrt_skip_example
    op: torch.rsqrt
    skip_if:
      - expr: cfg.test_device.type == 'spyre' and dtype_str == 'fp16'
        reason: Spyre fp16 rsqrt not supported yet
    inputs:
      - tensor: {shape: [64, 1024], init: rand}

  - name: getitem_ellipsis_none_fullslice
    op: torch.getitem
    inputs:
      - tensor:
          shape: [32, 5760]
          init: rand
      - py: (Ellipsis, None, slice(None, None, None))
```

---
## Metrics
Initial (functional) metrics:

- Pass rate per model and per-op.
- Coverage (# distinct ops and input shapes per model; % of enabled opFuncs).
- Deduplication efficiency (skipped duplicates vs unique executed cases).
- Stability trend across nightly runs (failures/new regressions).
- Runtime per case and total (optional for capacity planning).

Future (performance) metrics if enabled:

- Regression detection across releases (op latency deltas).
- Impact of backend optimizations.
- Comparison against baseline (CPU or another backend).

---
## Dependencies

- Existing internal scripts for op & input extraction.
- pytest runner (markers defined in pytest.ini). [docs.pytest.org]
- torch for CPU reference and .pt loading (baseline correctness). [docs.pytorch.org]
