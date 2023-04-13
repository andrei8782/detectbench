from pathlib import Path

# from probeinterface import Probe
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import spikeinterface as si
from IPython.display import display
from ipywidgets import FloatProgress, Layout
from numpy.typing import NDArray
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

from detectrun import (
    filter_ndarray_values,
    run_detection,
    validate_detection_config,
)

import os
import hashlib
import json
from collections import defaultdict


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def safe_percentage(part, total):
    if total == 0:
        return 0
    else:
        return 100 * part / total


def get_benchmark_file_path(
    recording_name: str,
    detection_method_name: str,
    detection_kwargs: Dict,
    search_radius: float,
    search_jitter: float,
) -> Path:
    def hash_dict(dictionary):
        dict_str = json.dumps(dictionary, sort_keys=True)
        return hashlib.sha256(dict_str.encode()).hexdigest()

    # numpy arrays are not hashable, so we need to remove them from the dict
    benchmark_kwargs = {
        "search_radius": search_radius,
        "search_jitter": search_jitter,
        **filter_ndarray_values(detection_kwargs),
    }
    file_name = hash_dict(benchmark_kwargs)

    file_path = Path(f"{recording_name}/{detection_method_name}/{file_name}")
    file_path = file_path.with_suffix(".npy")
    return file_path


def get_channels_in_radius(
    probe, radius: float, ch_ind: float
) -> Dict[int, float]:
    ch_locs = np.array(probe.contact_positions)
    target_loc = ch_locs[ch_ind]

    # Calculate squared Euclidean distances
    squared_distances = np.sum((ch_locs - target_loc) ** 2, axis=1)

    # Find channels within the specified radius
    within_radius = np.where(squared_distances <= radius ** 2)[0]

    # Calculate actual distances for channels within the radius
    distances = np.sqrt(squared_distances[within_radius])

    # Create a dictionary with channel indices and distances
    chs = dict(zip(within_radius, distances))

    return chs


def run_benchmark(
    recording: si.core.BaseRecording,
    extr_ch_to_units_trains: Dict[int, NDArray],
    detection_method: Callable,
    detection_kwargs: Optional[Dict] = {},
    detection_config_kwargs: Optional[Dict] = {},
    detection_job_kwargs: Optional[Dict] = {},
    search_radius: Optional[float] = None,
    search_jitter: Optional[float] = None,
    progress_bar: Optional[bool] = False,
    verbose: Optional[bool] = False,
    save_on_disk: Optional[bool] = False,
    folder_path: Optional[Path] = None,
    load_if_exists: Optional[bool] = None,
    overwrite: Optional[bool] = False,
    snr_per_unit_group: Optional[Dict[int, Tuple[List[int], float]]] = None,
) -> Dict[str, float]:
    supported_detections = [detect_peaks]
    if detection_method not in supported_detections:
        raise ValueError(
            f"Detection method {detection_method} not supported. Available methods: {supported_detections}"
        )

    if search_radius is None or search_jitter is None:
        raise NotImplementedError(
            "Automated computation of search params not yet implemented."
        )

    if load_if_exists and not overwrite:
        if folder_path is None:
            raise ValueError(
                "If load_if_exists is True, folder_path must be specified."
            )
        recording_name = Path(recording._kwargs["file_path"]).stem
        file_name = get_benchmark_file_path(
            recording_name,
            detection_method.__name__,
            detection_kwargs,
            search_radius,
            search_jitter,
        )
        file_path = folder_path / file_name
        if file_path.exists():
            if verbose:
                print(f"Loading benchmark results from {file_path}")
            benchmarking_log = np.load(file_path, allow_pickle=True).item()
            if verbose:
                print("Benchmark results loaded")
            return benchmarking_log
        else:
            if verbose:
                print(f"Benchmark results not found at {file_path}")

    validate_detection_config(detection_config_kwargs)

    sampling_frequency = recording.get_sampling_frequency()
    _gt_peaks = sum(
        [
            len(extr_ch_to_units_trains[unit])
            for unit in extr_ch_to_units_trains.keys()
        ]
    )
    _total_gt_peaks = _gt_peaks
    if progress_bar:
        bar = FloatProgress(
            min=0,
            max=_total_gt_peaks,
            bar_style="info",
            layout=Layout(width="99%"),
        )
        display(bar)

    peaks = run_detection(
        recording=recording,
        detection_method=detection_method,
        detection_kwargs=detection_kwargs,
        job_kwargs=detection_job_kwargs,
        **detection_config_kwargs,
    )
    matched_gt_peaks = 0
    matched_detected_peaks = 0
    duplicate_detected_peaks = 0
    already_marked_detected_peaks = 0
    equal_t_same_channel = 0
    marked = np.full(peaks.size, False)

    # Initialize group-specific metric dictionaries
    group_recall = defaultdict(float)
    group_precision = defaultdict(float)
    group_accuracy = defaultdict(float)
    group_snr = defaultdict()

    for extremum_ch, gt_units_group in extr_ch_to_units_trains.items():
        # Initialize group-specific counters
        group_matched_gt_peaks = 0
        group_matched_detected_peaks = 0
        # group_duplicate_detected_peaks = 0

        gt_peaks_t = gt_units_group / sampling_frequency
        chs = get_channels_in_radius(
            probe=recording.get_probe(),
            radius=search_radius,
            ch_ind=extremum_ch,
        )
        ch_inds = list(chs.keys())
        peaks_chs_filter = np.isin(peaks["channel_index"], ch_inds)
        peaks_subset = peaks[peaks_chs_filter]
        detected_peaks_t = peaks_subset["sample_ind"] / sampling_frequency
        detected_peaks_ch = peaks_subset["channel_index"]
        detected_peaks_inds = np.where(peaks_chs_filter)[0]
        for gt_peak_t in gt_peaks_t:
            dt = np.abs(detected_peaks_t - gt_peak_t)
            within_jitter = np.where(dt <= search_jitter)
            matched_chs = np.unique(detected_peaks_ch[within_jitter])
            if matched_chs.size > 0:
                matched_gt_peaks += 1
                group_matched_gt_peaks += 1
                if matched_chs.size > 1:
                    duplicate_detected_peaks += matched_chs.size - 1
                    # group_duplicate_detected_peaks += matched_chs.size - 1
            for matched_ch in matched_chs:
                matched_ch_filter = np.where(
                    detected_peaks_ch[within_jitter] == matched_ch
                )
                ch_filtered_dt = dt[within_jitter][matched_ch_filter]
                dt_min_inds = np.where(
                    dt[within_jitter] == np.min(ch_filtered_dt)
                )
                dt_min_inds_ch_filtered = dt_min_inds[0][
                    np.where(
                        detected_peaks_ch[within_jitter][dt_min_inds]
                        == matched_ch
                    )
                ]
                if dt_min_inds_ch_filtered.size > 1:
                    equal_t_same_channel += 1
                for min_ind in dt_min_inds_ch_filtered:
                    dt_min_ind = within_jitter[0][min_ind]
                    if not marked[detected_peaks_inds[dt_min_ind]]:
                        break
                assert detected_peaks_ch[dt_min_ind] == matched_ch
                assert (
                    peaks["sample_ind"][detected_peaks_inds[dt_min_ind]]
                    / recording.get_sampling_frequency()
                    == detected_peaks_t[dt_min_ind]
                )
                assert (
                    peaks["channel_index"][detected_peaks_inds[dt_min_ind]]
                    == detected_peaks_ch[dt_min_ind]
                )
                if marked[detected_peaks_inds[dt_min_ind]]:
                    already_marked_detected_peaks += 1
                marked[detected_peaks_inds[dt_min_ind]] = True
                matched_detected_peaks += 1
                group_matched_detected_peaks += 1
        if progress_bar:
            bar.value += len(gt_peaks_t)

        # Compute group-specific metrics
        group_recall[extremum_ch] = safe_divide(
            group_matched_gt_peaks, len(gt_units_group)
        )
        group_precision[extremum_ch] = safe_divide(
            group_matched_gt_peaks,
            group_matched_gt_peaks
            + (peaks_subset.size - group_matched_detected_peaks),
        )
        group_accuracy[extremum_ch] = safe_divide(
            group_matched_gt_peaks,
            len(gt_units_group)
            + (peaks_subset.size - group_matched_detected_peaks),
        )

        if snr_per_unit_group:
            group_snr[extremum_ch] = snr_per_unit_group[extremum_ch]

    if verbose:
        print(filter_ndarray_values(detection_kwargs))
        print(
            f"""
gt peaks:                       {_gt_peaks},
detected peaks:                 {len(peaks)} -- {safe_percentage(len(peaks), _gt_peaks):.2f}% of gt peaks,
matched gt peaks:               {matched_gt_peaks} -- {safe_percentage(matched_gt_peaks, _gt_peaks):.2f}% of gt peaks,
matched detected peaks:         {matched_detected_peaks} -- {safe_percentage(matched_detected_peaks, len(peaks)):.2f}% of detected peaks,
duplicate detected peaks:       {duplicate_detected_peaks} -- {safe_percentage(duplicate_detected_peaks, matched_detected_peaks):.2f}% of matched detected peaks,
unmatched detected peaks:       {len(peaks) - matched_detected_peaks} -- {safe_percentage(len(peaks) - matched_detected_peaks, len(peaks)):.2f}% of detected peaks,
already matched detected peaks: {already_marked_detected_peaks} -- {safe_percentage(already_marked_detected_peaks, matched_detected_peaks):.2f}% of matched detected peaks,
equal peak times same channel:  {equal_t_same_channel}"""
        )

    recall = safe_divide(matched_gt_peaks, _gt_peaks)
    precision = safe_divide(
        matched_gt_peaks,
        matched_gt_peaks + (peaks.size - matched_detected_peaks),
    )
    accuracy = safe_divide(
        matched_gt_peaks, _gt_peaks + (peaks.size - matched_detected_peaks)
    )

    if verbose:
        print(
            f"""
accuracy:  {accuracy:.2f},
precision: {precision:.2f},
recall:    {recall:.2f}
----------\n"""
        )
    benchmarking_log = dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        group_accuracy=dict(group_accuracy),
        group_precision=dict(group_precision),
        group_recall=dict(group_recall),
        group_snr=dict(group_snr),
    )

    if save_on_disk:
        if folder_path is None:
            raise ValueError(
                "folder_path must be specified if save_on_disk is True"
            )
        recording_name = Path(recording._kwargs["file_path"]).stem
        file_name = get_benchmark_file_path(
            recording_name,
            detection_method.__name__,
            detection_kwargs,
            search_radius,
            search_jitter,
        )
        file_path = folder_path / file_name
        if file_path.exists() and overwrite:
            if verbose:
                print(f"Overwriting benchmark results at {file_path}")
        elif not file_path.exists():
            if verbose:
                print(f"Saving benchmark results to {file_path}")
        elif file_path.exists() and not overwrite:
            raise FileExistsError(
                f"Benchmark results already exist at {file_path}"
            )
        if not file_path.parent.is_dir():
            os.makedirs(file_path.parent, exist_ok=True)
        np.save(file_path, benchmarking_log)
        if verbose:
            print("Benchmark results saved")

    return benchmarking_log
