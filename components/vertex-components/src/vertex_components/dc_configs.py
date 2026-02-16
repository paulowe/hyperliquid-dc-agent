from kfp.v2.dsl import component
from typing import NamedTuple

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
def get_dc_configs(
    thresholds: list,
    precision: int = 3
) -> NamedTuple(
    "Outputs",
    [
        ("indices", list),
        ("dc_thresholds_list", list),
        ("dc_table_list", list),
        ("x_table_list", list),
        ("exp_name_list", list),
        ("primary_thr_list", list),
        ("thr_str_list", list),
    ],
):
    def fmt(x: float) -> str:
        return f"{float(x):.{precision}f}"

    items = []
    for entry in thresholds:
        if isinstance(entry, (int, float)):
            down = up = float(entry)
        else:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError("Each threshold must be a float or 2-element (down, up).")
            down, up = float(entry[0]), float(entry[1])
            if down > up:
                down, up = up, down

        s_down, s_up = fmt(down), fmt(up)
        if down == up:
            thr_str  = s_down
            thr_safe = s_down.replace(".", "p")
            items.append({
                "dc_thresholds": [down],
                "dc_table":     f"dc_events_threshold_{thr_safe}",
                "x_table":      f"X_{thr_safe}",
                "exp_name":     f"dc_detection_threshold_{thr_str}",
                "primary_thr":  down,
                "thr_str":      thr_str,
            })
        else:
            thr_str  = f"{s_down}_{s_up}"
            thr_safe = f"{s_down.replace('.', 'p')}_{s_up.replace('.', 'p')}"
            items.append({
                "dc_thresholds": [down, up],
                "dc_table":     f"dc_events_threshold_{thr_safe}",
                "x_table":      f"X_{thr_safe}",
                "exp_name":     f"dc_detection_threshold_{thr_str}",
                "primary_thr":  up,
                "thr_str":      thr_str,
            })

    indices = list(range(len(items)))
    dc_thresholds_list = [it["dc_thresholds"] for it in items]
    dc_table_list      = [it["dc_table"]      for it in items]
    x_table_list       = [it["x_table"]       for it in items]
    exp_name_list      = [it["exp_name"]      for it in items]
    primary_thr_list   = [it["primary_thr"]   for it in items]
    thr_str_list       = [it["thr_str"]       for it in items]

    return (indices, dc_thresholds_list, dc_table_list,
            x_table_list, exp_name_list, primary_thr_list, thr_str_list)