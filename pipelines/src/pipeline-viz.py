import json
from graphviz import Digraph

with open("training.json") as f:
    pipeline = json.load(f)

dot = Digraph(comment="KFP DAG", format="png")
# dot.attr(rankdir="LR", fontsize="12", nodesep="0.4", ranksep="0.8")

components = pipeline["pipelineSpec"]["components"]

LABEL_MAP = {
    "bq-query-to-table": "Save Extracted Ticks dataset to BQ",
    "extract-bq-to-dataset": "Extract raw ticks to KFP managed storage",
    "directional-change-detector": "Directional Change Detection",
    "feature-engineering": "Feature Engineer DC Indicators",
    "tf-data-splitter": "Train/Validation/Test Split",
    "custom-vae-train-job": "Train VAE",
    "custom-forecast-train-job": "Train Forecast",
    "custom-train-job": "Train Latent Forecast",
    "lookup-model": "Lookup Past Latent Forecast Model",
    "for-loop-2": "Directional Change Analysis (ParallelFor)",
}


def human_label(name: str) -> str:
    return LABEL_MAP.get(name, name.replace("-", " ").title())


def traverse_dag(dag, components):
    """Traverse a DAG and draw tasks, recursively handling ParallelFor."""
    for task_name, task in dag.get("tasks", {}).items():
        label = task.get("taskInfo", {}).get("name", human_label(task_name))
        comp_ref = task.get("componentRef", {}).get("name")

        # ✅ Handle nested DAG (ParallelFor)
        if comp_ref and comp_ref in components and "dag" in components[comp_ref]:
            nested_dag = components[comp_ref]["dag"]

            # 🎯 Create a cluster for loop body
            with dot.subgraph(name=f"cluster_{task_name}") as sub:
                sub.attr(label=human_label(task_name), style="dashed", color="gray")

                # Draw nested tasks INSIDE the cluster
                for inner_task_name, inner_task in nested_dag["tasks"].items():
                    inner_label = inner_task.get("taskInfo", {}).get(
                        "name", human_label(inner_task_name)
                    )
                    sub.node(inner_task_name, inner_label)
                    for dep in inner_task.get("dependentTasks", []):
                        sub.edge(dep, inner_task_name)

            # 🚀 Connect parent DAG into and out of cluster
            for dep in task.get("dependentTasks", []):
                for entry in find_entry_tasks(nested_dag):
                    dot.edge(dep, entry)

            downstreams = find_downstream_tasks(dag, task_name)
            for exit_node in find_exit_tasks(nested_dag):
                for downstream in downstreams:
                    dot.edge(exit_node, downstream)

        else:
            # 🟢 Regular node
            dot.node(task_name, label)
            for dep in task.get("dependentTasks", []):
                dot.edge(dep, task_name)


def find_entry_tasks(dag):
    return [t for t, d in dag["tasks"].items() if not d.get("dependentTasks")]


def find_exit_tasks(dag):
    all_tasks = set(dag["tasks"].keys())
    all_deps = {dep for d in dag["tasks"].values() for dep in d.get("dependentTasks", [])}
    return list(all_tasks - all_deps)


def find_downstream_tasks(dag, task_name):
    return [
        t
        for t, details in dag.get("tasks", {}).items()
        if task_name in details.get("dependentTasks", [])
    ]


# 🚀 Kick off traversal from the root DAG
root_dag = pipeline["pipelineSpec"]["root"]["dag"]
traverse_dag(root_dag, components)

dot.render("pipeline_dag", view=False)
print("✅ DAG image generated: pipeline_dag.png (with proper ParallelFor grouping)")
