# Dataflow Components

## Directional Change Streaming Pipeline

The module `dataflow_components.directional_change_streaming` hosts an Apache Beam
streaming pipeline that consumes tick data from Pub/Sub and emits directional
change events to another Pub/Sub topic. The logic incrementally maintains the
threshold state per instrument using Beam stateful processing, allowing the job
to operate on unbounded data without materialising large windows.

### Launching from Kubeflow Pipelines

The KFP pipeline defined in `dataflow_components.job.directional_change_streaming_pipeline`
wraps the Beam job in `DataflowPythonJobOp`. Provide the Pub/Sub resources and
threshold configuration when constructing the pipeline:

```python
from dataflow_components import job

job.directional_change_streaming_pipeline(
    input_subscription="projects/<proj>/subscriptions/ticks-stream",
    output_topic="projects/<proj>/topics/dc-events",
    thresholds="0.003,0.005:0.006",
)
```

The comma-separated `thresholds` string supports single values (applied to both
upward and downward movement) or `down:up` pairs for asymmetric detection. Each
value becomes a `--threshold` argument for the Beam job.

### Direct Runner Invocation

Run the pipeline module locally for smoke testing by supplying the required
arguments:

```bash
python -m dataflow_components.directional_change_streaming \
  --input_subscription=projects/<proj>/subscriptions/ticks-stream \
  --output_topic=projects/<proj>/topics/dc-events \
  --threshold=0.003 \
  --threshold=0.005:0.007
```

The job expects JSON messages with at least `symbol`, `price`, and `timestamp`
fields. The timestamp can be ISO-8601 (with `Z` support) or a Unix epoch.

When deployed on Dataflow the pipeline enables the streaming engine and runner
V2, and relies on Dataflow autoscaling to adjust worker concurrency.
