from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from numpy.typing import NDArray
import spikeinterface.qualitymetrics as qm


def load_recording(file_path: Path) -> si.core.BaseRecording:
    if file_path.suffix == ".h5" and "mearec" in file_path.name:
        recording = se.MEArecRecordingExtractor(file_path)
    else:
        raise ValueError(f"Recording extractor '{file_path}' not supported.")
    return recording


def load_sorting(file_path: Path) -> si.core.BaseSorting:
    if file_path.suffix == ".h5" and "mearec" in file_path.name:
        sorting = se.MEArecSortingExtractor(file_path)
    else:
        raise ValueError(f"Sorting extractor '{file_path}' not supported.")
    return sorting


def preprocess_recording(
    recording: si.core.BaseRecording,
    save_on_disk: Optional[bool] = False,
    folder_path: Optional[Path] = None,
    load_if_exists: Optional[bool] = None,
    overwrite: Optional[bool] = False,
    **kwargs,
) -> si.core.BaseRecording:
    pass


def get_waveform_extractor(
    recording: si.core.BaseRecording,
    sorting: si.core.BaseSorting,
    job_kwargs: Optional[Dict] = {},
) -> si.core.WaveformExtractor:
    # data_path might break when loading a preprocessed recording due to
    # different file naming compared to the original recording.
    data_path = Path(recording._kwargs["file_path"])
    wf_folder = Path(f"waveforms/{data_path.stem}")
    if recording.is_filtered():
        wf_folder = wf_folder / "filtered"
    else:
        wf_folder = wf_folder / "raw"
    if wf_folder.exists():
        we = si.load_waveforms(wf_folder)
    else:
        we = si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=wf_folder,
            overwrite=False,
            **job_kwargs,
        )
    return we


def compute_snrs_per_unit_group(
    recording: si.core.BaseRecording,
    sorting: si.core.BaseSorting,
    extremum_ch_to_units_inds: Dict[int, List[int]],
    job_kwargs: Optional[Dict] = {},
) -> Dict[int, Tuple[List[int], float]]:
    # returns a dict with keys as extremum channels and values as tuples of
    # list of unit inds and their mean snr
    we = get_waveform_extractor(
        recording=recording, sorting=sorting, job_kwargs=job_kwargs
    )

    snrs_per_unit = qm.compute_snrs(we)
    # convert keys of snrs from id to integer index (from 0)
    snrs_per_unit = {
        np.where(sorting.get_unit_ids() == k)[0][0]: v
        for k, v in snrs_per_unit.items()
    }

    snrs_per_unit_group = {}
    for ch, unit_inds in extremum_ch_to_units_inds.items():
        snrs_per_unit_group[ch] = (
            unit_inds,
            np.mean([snrs_per_unit[i] for i in unit_inds]),
        )

    return snrs_per_unit_group


def compute_extremum_channels(
    recording: si.core.BaseRecording,
    sorting: si.core.BaseSorting,
    job_kwargs: Optional[Dict] = {},
) -> Dict[int, List[int]]:
    we = get_waveform_extractor(
        recording=recording, sorting=sorting, job_kwargs=job_kwargs
    )

    extremum_channels_inds = si.core.get_template_extremum_channel(
        waveform_extractor=we, peak_sign="neg", outputs="index"
    )

    unique_extremum_chs = np.unique(list(extremum_channels_inds.values()))

    # Map unit inds to their mutual extremum channel
    extremum_ch_to_units_inds = {}
    for ch in unique_extremum_chs:
        extremum_ch_to_units_inds[ch] = [
            np.where(sorting.get_unit_ids() == k)[0][0]
            for k, v in extremum_channels_inds.items()
            if v == ch
        ]

    return extremum_ch_to_units_inds


def merge_units_by_extremum_ch(
    recording: si.core.BaseRecording,
    sorting: si.core.BaseSorting,
    job_kwargs: Optional[Dict] = {},
) -> Dict[int, NDArray]:
    extr_ch_to_units_inds = compute_extremum_channels(
        recording=recording, sorting=sorting, job_kwargs=job_kwargs
    )

    # Map a group of gt units to their mutual extremum channel
    sorting_gt_groups = {}
    for ch, unit_inds in extr_ch_to_units_inds.items():
        gt_units_spike_train = [
            sorting.get_unit_spike_train(
                unit_id=sorting.get_unit_ids()[unit_ind]
            )
            for unit_ind in unit_inds
        ]
        merged_units = np.concatenate(gt_units_spike_train)
        # Sort to match closest in time first
        merged_units.sort(kind="stable")
        sorting_gt_groups[ch] = merged_units

    return sorting_gt_groups, extr_ch_to_units_inds
