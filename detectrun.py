import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import spikeinterface as si
from numpy.typing import NDArray
from spikeinterface.sortingcomponents.peak_pipeline import base_peak_dtype


def validate_detection_config(detection_config_kwargs: Dict) -> bool:
    detection_config = dict(
        verbose=False,
        save_on_disk=False,
        folder_path=None,
        load_if_exists=None,
        overwrite=False,
    )

    for key, value in detection_config_kwargs.items():
        if key not in detection_config.keys():
            raise ValueError("detection_config kwarg {key} not supported.")


def filter_ndarray_values(dictionary):
    return {
        k: v for k, v in dictionary.items() if not isinstance(v, np.ndarray)
    }


def get_detection_file_path(
    recording_name: str,
    detection_method_name: str,
    detection_kwargs: Optional[Dict] = {},
) -> Path:
    def hash_dict(dictionary):
        dict_str = json.dumps(dictionary, sort_keys=True)
        return hashlib.sha256(dict_str.encode()).hexdigest()

    # numpy arrays are not hashable, so we need to remove them from the dict
    file_name = hash_dict(filter_ndarray_values(detection_kwargs))
    file_path = Path(f"{recording_name}/{detection_method_name}/{file_name}")
    file_path = file_path.with_suffix(".npy")
    return file_path


def run_detection(
    recording: si.core.BaseRecording,
    detection_method: Callable,
    detection_kwargs: Optional[Dict] = {},
    verbose: Optional[bool] = False,
    save_on_disk: Optional[bool] = False,
    folder_path: Optional[Path] = None,
    load_if_exists: Optional[bool] = None,
    overwrite: Optional[bool] = False,
    job_kwargs: Optional[Dict] = {},
) -> NDArray[base_peak_dtype]:
    detection_method_name = detection_method.__name__
    if load_if_exists and not overwrite:
        if folder_path is None:
            raise ValueError("Please provide a folder path.")
        else:
            recording_name = Path(recording._kwargs["file_path"]).stem
            file_name = get_detection_file_path(
                recording_name, detection_method_name, detection_kwargs
            )
            file_path = folder_path / file_name
            if file_path.exists():
                if verbose:
                    print(f"Loading detection results from {file_path}")
                peaks = np.load(file_path)
                if verbose:
                    print(f"{peaks.size} peaks loaded")
                return peaks
            else:
                if verbose:
                    print(f"Detection results not found at {file_path}")

    if detection_method_name == "detect_peaks":
        peaks = detection_method(
            recording=recording, **detection_kwargs, **job_kwargs
        )
    else:
        peaks = detection_method(recording, **detection_kwargs)

    if verbose:
        # Log the detection args too
        print(f"{peaks.size} peaks detected")
    if save_on_disk:
        if folder_path is None:
            raise ValueError("Please provide a folder path.")
        else:
            recording_name = Path(recording._kwargs["file_path"]).stem
            file_name = get_detection_file_path(
                recording_name, detection_method_name, detection_kwargs
            )
            file_path = folder_path / file_name
            if file_path.exists() and overwrite:
                if verbose:
                    print(f"Overwriting detection results at {file_path}")
            elif not file_path.exists():
                if verbose:
                    print(f"Saving detection results to {file_path}")
            elif file_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Detection results already exist at {file_path}"
                )
            if not file_path.parent.is_dir():
                os.makedirs(file_path.parent, exist_ok=True)
            np.save(file_path, peaks)
            if verbose:
                print(f"{peaks.size} peaks saved")
    return peaks
