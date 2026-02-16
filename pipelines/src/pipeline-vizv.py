import json
from graphviz import Digraph

with open("training.json") as f:
    pipeline = json.load(f)

dot = Digraph(comment="KFP DAG", format="png")

# dot.attr(rankdir='LR', fontsize='12', nodesep='0.4', ranksep='0.8')

def traverse_dag(dag, components, prefix=""):
    """Recursively add all tasks and nested DAGs to the graph."""
    for task_name, task in dag.get("tasks", {}).items():
        label = task.get("taskInfo", {}).get("name", task_name)
        node_id = f"{prefix}{task_name}"
        dot.node(node_id, label)

        for dep in task.get("dependentTasks", []):
            dot.edge(f"{prefix}{dep}", node_id)

        # If this task refers to a component with its own DAG (like ParallelFor)
        comp_ref = task.get("componentRef", {}).get("name")
        if comp_ref and comp_ref in components:
            comp_def = components[comp_ref]
            if "dag" in comp_def:
                traverse_dag(comp_def["dag"], components, prefix=node_id + "/")

# Kick off from the root DAG
components = pipeline["pipelineSpec"]["components"]
root_dag = pipeline["pipelineSpec"]["root"]["dag"]
traverse_dag(root_dag, components)

dot.render("pipeline_dag", view=False)
print("✅ DAG image generated including tasks inside ParallelFor")
