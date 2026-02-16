import json
from graphviz import Digraph

with open("training.json") as f:
    pipeline = json.load(f)

dot = Digraph(comment="KFP DAG", format="png")
# dot.attr(rankdir="LR", fontsize="12", nodesep="0.4", ranksep="0.8")

components = pipeline["pipelineSpec"]["components"]

def traverse_dag(dag, components, cluster=None):
    """Draw DAG recursively, grouping nested DAGs into clusters."""
    for task_name, task in dag.get("tasks", {}).items():
        label = task.get("taskInfo", {}).get("name", task_name)
        comp_ref = task.get("componentRef", {}).get("name")

        # If task is a ParallelFor (nested DAG)
        if comp_ref and comp_ref in components and "dag" in components[comp_ref]:
            nested_dag = components[comp_ref]["dag"]
            # Create a subgraph cluster
            with dot.subgraph(name=f"cluster_{task_name}") as sub:
                sub.attr(label=f"{label} (loop)", style="dashed", color="gray")
                traverse_dag(nested_dag, components, cluster=sub)

            # Now wire edges: upstream → first tasks inside
            for dep in task.get("dependentTasks", []):
                for entry in find_entry_tasks(nested_dag):
                    dot.edge(dep, entry)

            # and last tasks inside → downstream of for-loop
            downstreams = find_downstream_tasks(dag, task_name)
            for exit_node in find_exit_tasks(nested_dag):
                for down in downstreams:
                    dot.edge(exit_node, down)

        else:
            # Normal node
            dot.node(task_name, label)
            for dep in task.get("dependentTasks", []):
                dot.edge(dep, task_name)


def find_entry_tasks(dag):
    """Tasks in a nested DAG with no dependencies (entry points)."""
    entries = []
    for t, details in dag["tasks"].items():
        if not details.get("dependentTasks"):
            entries.append(t)
    return entries

def find_exit_tasks(dag):
    """Tasks in a nested DAG with no children (exit points)."""
    all_tasks = set(dag["tasks"].keys())
    all_deps = set(d for task in dag["tasks"].values() for d in task.get("dependentTasks", []))
    exits = list(all_tasks - all_deps)
    return exits

def find_downstream_tasks(dag, task_name):
    """Find tasks that depend on the given task in the parent DAG."""
    downstreams = []
    for t, details in dag.get("tasks", {}).items():
        if task_name in details.get("dependentTasks", []):
            downstreams.append(t)
    return downstreams


# Kick off from root DAG
root_dag = pipeline["pipelineSpec"]["root"]["dag"]
traverse_dag(root_dag, components)

dot.render("pipeline_dag", view=False)
print("✅ DAG with ParallelFor grouped in clusters saved: pipeline_dag.pdf")
