# Tests for the DC feature ablation pipeline compilation.

import json
import os
import tempfile

import pytest
from kfp.v2 import compiler


def test_ablation_pipeline_compiles():
    """The ablation pipeline must compile to a valid JSON IR without errors."""
    from pipelines.tensorflow.training.pipeline_ablation import ablation_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "ablation.json")
        compiler.Compiler().compile(
            pipeline_func=ablation_pipeline,
            package_path=output_path,
            type_check=False,
        )
        # Verify the file was created and is valid JSON
        assert os.path.exists(output_path), "Compiled pipeline JSON was not created"
        with open(output_path) as f:
            pipeline_ir = json.load(f)
        # Basic structure checks
        assert "pipelineSpec" in pipeline_ir or "root" in pipeline_ir, (
            "Pipeline IR is missing expected top-level keys"
        )


def test_ablation_pipeline_has_comparison_step():
    """The compiled pipeline should contain a compare_forecasts component."""
    from pipelines.tensorflow.training.pipeline_ablation import ablation_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "ablation.json")
        compiler.Compiler().compile(
            pipeline_func=ablation_pipeline,
            package_path=output_path,
            type_check=False,
        )
        with open(output_path) as f:
            pipeline_ir = json.load(f)
        # Serialize to string to search for component name
        ir_str = json.dumps(pipeline_ir)
        assert "compare-forecasts" in ir_str or "compare_forecasts" in ir_str, (
            "Pipeline IR should reference the compare_forecasts component"
        )
