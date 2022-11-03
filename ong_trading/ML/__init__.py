from __future__ import annotations

import os


def get_model_path(model_name: str, extra: str | bool = False) -> str:
    """
    Gets the model path for use as input of keras.model.load_model and save_model
    Model is saved in <thisdir>/RL/results/models/<model_name>/<extra>
    :param model_name: name of the model,
    :param extra: if True, extra="train". If False extra="no_train". If a string, extra is that string
    :return: a string with the path
    """
    """"""
    this_file_path = os.path.dirname(__file__)
    if isinstance(extra, bool):
        extra_path = "train" if extra else "no_train"
    elif isinstance(extra, str):
        extra_path = extra
    else:
        extra_path = ""
    model_path = os.path.join(this_file_path, "RL", "results", "models", model_name,
                              extra_path)
    return model_path
