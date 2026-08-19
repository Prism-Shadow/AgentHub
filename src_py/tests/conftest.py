# Copyright 2025 Prism Shadow. and/or its affiliates
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

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Pin every test of one model to a single xdist worker.

    Providers rate-limit per client, so under `-n <N> --dist loadgroup` distinct models run
    in parallel while one model's tests stay serial on the worker that owns its group.
    """
    for item in items:
        callspec = getattr(item, "callspec", None)
        model = callspec.params.get("model") if callspec else None
        if model is not None:
            item.add_marker(pytest.mark.xdist_group(str(model)))
