import json
from graphviz import Digraph

# 📂 Load compiled pipeline spec
with open("training.json") as f:
    pipeline = json.load(f)

# 🎨 Stage colors
STAGE_COLORS = {
    "data": "#c6d9f1",   # soft blue
    "event": "#f9cb9c",  # soft orange
    "model": "#d9ead3",  # soft green
}

# 🧠 Global styling
dot = Digraph(comment="KFP DAG", format="png")
dot.attr(rankdir="TB", nodesep="0.5", ranksep="0.8")

components = pipeline["pipelineSpec"]["components"]

# ✅ Human-friendly labels
LABEL_MAP = {
    "bq-query-to-table": "Save Extracted Ticks dataset to BQ",
    "extract-bq-to-dataset": "Extract raw ticks to KFP managed storage",
    "directional-change-detector": "DC Detection",
    "feature-engineering": "Feature Engineer DC indicators",
    "tf-data-splitter": "Train/Validation/Test Split",
    "custom-vae-train-job": "Train VAE",
    "custom-forecast-train-job": "Train Forecast",
    "custom-train-job": "Train Latent Forecast",
    "lookup-model": "Lookup past latent forecast model",
    "for-loop-2": "",
}

# 📌 Stage mapping for color coding
STAGE_MAP = {
    "bq-query-to-table": "data",
    "extract-bq-to-dataset": "data",
    "directional-change-detector": "event",
    "feature-engineering": "event",
    "tf-data-splitter": "event",
    "lookup-model": "event",
    "custom-vae-train-job": "model",
    "custom-forecast-train-job": "model",
    "custom-train-job": "model",
}


def human_label(name: str) -> str:
    return LABEL_MAP.get(name, name.replace("-", " ").title())


def node_color(name: str) -> str:
    return STAGE_COLORS.get(STAGE_MAP.get(name, "event"), "#ffffff")


def traverse_dag(dag, components):
    """Recursively traverse DAG and draw tasks."""
    for task_name, task in dag.get("tasks", {}).items():
        label = task.get("taskInfo", {}).get("name", human_label(task_name))
        comp_ref = task.get("componentRef", {}).get("name")

        # 📦 Handle nested DAG (ParallelFor)
        if comp_ref and comp_ref in components and "dag" in components[comp_ref]:
            nested_dag = components[comp_ref]["dag"]
            with dot.subgraph(name=f"cluster_{task_name}") as sub:
                sub.attr(label="", style="dashed", color="gray", fontsize="14", margin="30")
                for inner_task_name, inner_task in nested_dag["tasks"].items():
                    inner_label = inner_task.get("taskInfo", {}).get(
                        "name", human_label(inner_task_name)
                    )
                    sub.node(inner_task_name, inner_label, style="filled", fillcolor=node_color(inner_task_name))
                    for dep in inner_task.get("dependentTasks", []):
                        sub.edge(dep, inner_task_name)

            for dep in task.get("dependentTasks", []):
                for entry in find_entry_tasks(nested_dag):
                    dot.edge(dep, entry)
            downstreams = find_downstream_tasks(dag, task_name)
            for exit_node in find_exit_tasks(nested_dag):
                for downstream in downstreams:
                    dot.edge(exit_node, downstream)
        else:
            dot.node(task_name, label, style="filled", fillcolor=node_color(task_name))
            for dep in task.get("dependentTasks", []):
                dot.edge(dep, task_name)


def find_entry_tasks(dag):
    return [t for t, d in dag["tasks"].items() if not d.get("dependentTasks")]


def find_exit_tasks(dag):
    all_tasks = set(dag["tasks"].keys())
    all_deps = {dep for d in dag["tasks"].values() for dep in d.get("dependentTasks", [])}
    return list(all_tasks - all_deps)


def find_downstream_tasks(dag, task_name):
    return [t for t, details in dag.get("tasks", {}).items() if task_name in details.get("dependentTasks", [])]


# 🚀 Build main DAG
root_dag = pipeline["pipelineSpec"]["root"]["dag"]
traverse_dag(root_dag, components)

# 📍 Invisible anchor to place legend below DAG
dot.node("legend_anchor", label="", shape="point", width="0", style="invis")
dot.edge("custom-train-job", "legend_anchor", style="invis")

# 🪄 Horizontal legend row in same rank block
with dot.subgraph() as legend:
    legend.attr(rank="same")

    legend.node("legend_data", "Data Ingestion",
                shape="box", style="filled", fillcolor=STAGE_COLORS["data"],
                width="1.8", height="0.4", fontsize="10")
    legend.node("legend_event", "Event Detection",
                shape="box", style="filled", fillcolor=STAGE_COLORS["event"],
                width="1.8", height="0.4", fontsize="10")
    legend.node("legend_model", "Model Training",
                shape="box", style="filled", fillcolor=STAGE_COLORS["model"],
                width="1.8", height="0.4", fontsize="10")

# ✅ Invisible edges force them side by side
dot.edge("legend_data", "legend_event", style="invis")
dot.edge("legend_event", "legend_model", style="invis")

# 📍 Attach legend below DAG
dot.edge("legend_anchor", "legend_data", style="invis")

# 📦 Export final figure
dot.format = "png"
dot.render("pipeline_dag_horizontal_legend", view=False)

print("✅ DAG image generated: pipeline_dag_horizontal_legend.png")
