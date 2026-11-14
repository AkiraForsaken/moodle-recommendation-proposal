"""Phase 3 estimator: the deep KT model trained on this course's answers.

A thin adapter. All model code lives in ``knowledge_tracing``; this module only
runs its export for the Moodle-format dataset and renames the result to the
knowledge-state contract. PyTorch and pyKT are imported only when it is called.
"""

from pathlib import Path

import numpy as np

from . import config
from .data import ANSWER_EVENTS_FILE, ROSTER_FILE
from .schemas import KNOWLEDGE_STATE_COLUMNS, validate_states

ESTIMATOR_PHASE = 3


def to_contract(deep_states, mapping_version, source_population):
    """Rename the deep export's rows to the shared knowledge-state contract."""
    states = deep_states.copy()
    states["p_mastery"] = np.nan  # the deep model has no latent-mastery quantity
    states["estimator_phase"] = ESTIMATOR_PHASE
    states["mapping_version"] = mapping_version
    states["source_population"] = source_population
    return validate_states(states[KNOWLEDGE_STATE_COLUMNS])


def estimate_states(
    data_dir,
    work_dir,
    as_of=None,
    mapping_version=None,
    source_population=config.MOODLE_POPULATION,
    device=None,
):
    """Run the selected checkpoint over every roster learner and catalog KC."""
    from knowledge_tracing.export_states import export_states

    data_dir = Path(data_dir)
    roster = data_dir / ROSTER_FILE
    deep_states = export_states(
        input_path=data_dir / ANSWER_EVENTS_FILE,
        output_dir=Path(work_dir),
        dataset_name=config.DEEP_DATASET_NAME,
        learner_roster_path=roster if roster.is_file() else None,
        as_of=as_of,
        device=device,
    )
    return to_contract(deep_states, mapping_version, source_population)
