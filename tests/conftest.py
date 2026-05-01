"""Project-wide pytest configuration.

The ``gpu`` marker tags tests that require a CUDA device. Local runs
without a GPU skip those tests automatically with a helpful message
that points at the remote setup guide.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA-capable GPU. Skipped automatically when CUDA is absent.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    try:
        import torch  # noqa: PLC0415

        gpu_available = torch.cuda.is_available()
    except Exception:
        gpu_available = False

    if gpu_available:
        return

    skip_gpu = pytest.mark.skip(
        reason=(
            "No CUDA GPU available. Run on the remote 4xH100 host. "
            "See docs/remote-setup.md."
        )
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
