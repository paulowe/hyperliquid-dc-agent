# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import Input, component, Model


@component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-aiplatform==1.24.1",
    ],
)
def get_model_resource_name(
    model: Input[Model],
) -> str:
    """Extract resource name from a Model artifact.
    
    Args:
        model (Input[Model]): Model artifact
        
    Returns:
        str: Resource name of the model
    """
    return model.metadata.get("resourceName", "") 