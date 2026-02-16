from kfp.v2.dsl import component
import typing

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
        "typing",
    ],
)
def unpack_dc_config(config: dict) -> typing.NamedTuple('Out', [
    ('dc_table', str), ('x_table', str), ('exp_name', str), ('model_suffix', str),
    ('primary_thr', float), ('down_thr', float), ('up_thr', float),
    ('dc_thresholds', list), ('thr_str', str),
]):
    return (config['dc_table'], config['x_table'], config['exp_name'], config['model_suffix'],
            float(config['primary_thr']), float(config['down_thr']), float(config['up_thr']),
            config['dc_thresholds'], config['thr_str'])
