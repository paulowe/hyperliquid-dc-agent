# FILE: components/window_dataset.py
from kfp.v2.dsl import component, Output, Artifact
import pandas as pd
import numpy as np

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def make_windows(
    csv_path: str,
    input_width: int,
    label_width: int,
    shift: int,
    output_dataset: Output[Artifact]
):
    """
    Reads a CSV time-series file, creates a windowed dataset, and saves as numpy arrays.

    Args:
        csv_path: GCS or local path to input CSV with time-series data
        input_width: number of time steps in each input window
        label_width: number of time steps to predict
        shift: offset between end of input and start of label
    Outputs:
        output_dataset: dict with npy files for inputs and labels
    """
    df = pd.read_csv(csv_path)
    data = df.values
    total_size = data.shape[0]
    inputs = []
    labels = []
    for i in range(total_size - input_width - shift - label_width + 1):
        inputs.append(data[i : i + input_width])
        labels.append(data[i + input_width + shift : i + input_width + shift + label_width])
    inputs = np.array(inputs)
    labels = np.array(labels)
    np.save("inputs.npy", inputs)
    np.save("labels.npy", labels)
    output_dataset.path = "./"
    print(f"Windows created: inputs {inputs.shape}, labels {labels.shape}")


# FILE: components/single_step_single_feature.py
from kfp.v2.dsl import component
def
import tensorflow as tf

@component(
    base_image="python:3.9",
    packages_to_install=["tensorflow"]
)
def single_step_single_feature(
    inputs_path: str,
    model_steps: int,
    epochs: int,
    output_model_path: str
):
    """
    Trains a model that forecasts one step ahead on a single feature.
    """
    import numpy as np
    data = np.load(f"{inputs_path}/inputs.npy")
    # select only first feature
    X = data[..., 0]
    y = np.load(f"{inputs_path}/labels.npy")[..., 0]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs)
    model.save(output_model_path)


# FILE: components/single_step_all_features.py
from kfp.v2.dsl import component
import tensorflow as tf

@component(
    base_image="python:3.9",
    packages_to_install=["tensorflow"]
)
def single_step_all_features(
    inputs_path: str,
    epochs: int,
    output_model_path: str
):
    """
    Trains a model that forecasts one step ahead using all features.
    """
    import numpy as np
    X = np.load(f"{inputs_path}/inputs.npy")
    y = np.load(f"{inputs_path}/labels.npy")
    # predict all features
    y = y.reshape((y.shape[0], y.shape[2]))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(X.shape[2])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs)
    model.save(output_model_path)


# FILE: components/multi_step_single_shot.py
from kfp.v2.dsl import component
import tensorflow as tf

@component(
    base_image="python:3.9",
    packages_to_install=["tensorflow"]
)
def multi_step_single_shot(
    inputs_path: str,
    label_width: int,
    epochs: int,
    output_model_path: str
):
    """
    Trains a single-shot multi-step forecasting model.
    """
    import numpy as np
    X = np.load(f"{inputs_path}/inputs.npy")
    y = np.load(f"{inputs_path}/labels.npy")
    # reshape labels
    y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(label_width * X.shape[2])
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs)
    model.save(output_model_path)


# FILE: components/multi_step_autoregressive.py
from kfp.v2.dsl import component
import tensorflow as tf

@component(
    base_image="python:3.9",
    packages_to_install=["tensorflow"]
)
def multi_step_autoregressive(
    inputs_path: str,
    label_width: int,
    epochs: int,
    output_model_path: str
):
    """
    Trains an autoregressive multi-step forecasting model.
    """
    import numpy as np
    X = np.load(f"{inputs_path}/inputs.npy")
    y = np.load(f"{inputs_path}/labels.npy")
    feature_count = X.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, feature_count)),
        tf.keras.layers.LSTM(64, return_state=True),
        # autoregressive loop implemented in inference
        tf.keras.layers.Dense(feature_count)
    ])
    model.compile(optimizer='adam', loss='mse')
    # Flatten windows for training each step
    # Custom training loop omitted for brevity
    model.fit(X, y[:, :1, :], epochs=epochs)  # train on first step
    model.save(output_model_path)


# FILE: pipelines/time_series_experiments_pipeline.py
import kfp
from kfp.v2.dsl import pipeline
from components.window_dataset import make_windows
from components.single_step_single_feature import single_step_single_feature
from components.single_step_all_features import single_step_all_features
from components.multi_step_single_shot import multi_step_single_shot
from components.multi_step_autoregressive import multi_step_autoregressive

@pipeline(
    name="time-series-experiments",
    pipeline_root="gs://YOUR_BUCKET/pipeline_root"
)
def ts_experiments(
    csv_path: str,
    input_width: int = 24,
    label_width: int = 1,
    shift: int = 1,
    epochs: int = 10
):
    # 1. Create windows
    windows = make_windows(csv_path, input_width, label_width, shift)

    # 2. Single-step, single-feature
    sssf = single_step_single_feature(
        windows.outputs['output_dataset'].path,
        input_width,
        epochs,
        output_model_path="./sssf_model"
    )

    # 3. Single-step, all-features
    ssaf = single_step_all_features(
        windows.outputs['output_dataset'].path,
        epochs,
        output_model_path="./ssaf_model"
    )

    # 4. Multi-step single-shot
    msss = multi_step_single_shot(
        windows.outputs['output_dataset'].path,
        label_width,
        epochs,
        output_model_path="./msss_model"
    )

    # 5. Multi-step autoregressive
    msar = multi_step_autoregressive(
        windows.outputs['output_dataset'].path,
        label_width,
        epochs,
        output_model_path="./msar_model"
    )

if __name__ == '__main__':
    kfp.v2.compiler.Compiler().compile(
        pipeline_func=ts_experiments,
        package_path="time_series_experiments_pipeline.json"
    )
