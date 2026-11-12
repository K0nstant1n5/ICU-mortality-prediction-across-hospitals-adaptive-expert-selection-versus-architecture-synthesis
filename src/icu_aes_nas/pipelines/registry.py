"""Pipelines share the data layout, token bus, blocks, AES, NAS and evaluation. They differ in sources and stems.

Pipeline B (implemented): structured ICU time series from several databases; stems = linear | grud.
Pipeline A (to add):      MIMIC-IV structured + ECG + CXR. Add:
    * data: ECG/CXR linking (subject_id + time window) writing extra arrays next to X.npy,
    * core/stems.py: ECG and CXR stems emitting tokens with TOKEN_ECG / TOKEN_CXR and their own `time`,
    * core/network.py: concatenate modality tokens into the bus (fusion only at blocks with
      `can_fuse_modalities = True`; TCN/GRU act on EHR tokens only),
    * aes/router.py: pass the per-sample modality availability mask (`avail`).
Nothing in Pipeline B needs to change for that.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineInfo:
    name: str
    stems: tuple[str, ...]
    modalities: tuple[str, ...]


PIPELINES = {
    "B": PipelineInfo("B", stems=("linear", "grud"), modalities=("ehr",)),
}


def get_pipeline(name: str) -> PipelineInfo:
    if name not in PIPELINES:
        raise NotImplementedError(f"Pipeline {name} is not implemented yet (see pipelines/registry.py).")
    return PIPELINES[name]
