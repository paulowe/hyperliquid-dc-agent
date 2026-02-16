from kfp.v2.dsl import component, Input, Output, Dataset, Artifact
from typing import List, Sequence

@component(
    base_image="python:3.11",
    packages_to_install=[
        # Upgrade pip/setuptools so we get the latest PEP 517 hooks
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        # Force Cython < 3 so build‐time hooks don’t look for the removed API
        "cython<3.0.0",
        # Install PyYAML without isolation so it sees our Cython
        "--no-build-isolation",
        "pyyaml==5.4.1",
    ],
)
def pick_reference_scaler(
    scalers: Input[List[Artifact]],
    reference: Output[Artifact],
):
    import shutil
    import os
    src = scalers[0].path  # use first as reference
    shutil.copytree(src, reference.path, dirs_exist_ok=True)

# reference_scaler = pick_reference_scaler(scalers=scalers)