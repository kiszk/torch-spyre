# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest
import torch

from model_cases_loader import case_key, load_all_cases, to_pytest_params
from runner import RunConfig, run_case


def _collect_params(pytest_root: Path):
    items = load_all_cases(pytest_root)
    return to_pytest_params(items)


# NOTE: this runs at collection time; use pytestconfig.rootpath so paths are stable
def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames and "case" in metafunc.fixturenames:
        root = metafunc.config.rootpath
        params = _collect_params(root)
        metafunc.parametrize("model,case,defaults,source_path", params)


def test_model_ops(
    model,
    case,
    defaults,
    source_path,
    selected_models,
    dedupe_enabled,
    seen_case_keys,
    test_device_str,
    compile_backend,
):
    # 1) Model filtering (keeps your “only ops from granite3-speech” capability)
    if selected_models and model not in selected_models:
        pytest.skip(f"Filtered out by --model (selected={sorted(selected_models)})")

    # 2) Optional cross-model dedupe (do NOT dedupe at collection time!)
    if dedupe_enabled:
        k = case_key(case, defaults)
        if k in seen_case_keys:
            pytest.skip(
                "duplicate signature already tested (enable/disable via --no-dedupe)"
            )
        seen_case_keys.add(k)

    cfg = RunConfig(
        test_device=torch.device(test_device_str),
        compile_backend=compile_backend,
    )
    try:
        run_case(case, defaults, cfg)
    finally:
        torch._dynamo.reset()
