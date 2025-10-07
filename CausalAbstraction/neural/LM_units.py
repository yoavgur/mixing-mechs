"""
LM_units.py
===========
Helpers that bind the *core* component / featurizer abstractions from
`model_units.py` to language-model pipelines.  They let you refer to:

* A **ResidualStream** slice: hidden state of one or more token positions.
* An **AttentionHead** value: output for a single attention head.

All helpers inherit from :class:`model_units.AtomicModelUnit`, so they carry
the full featurizer + feature indexing machinery.
"""

import sys
from pathlib import Path
from typing import List, Union

sys.path.append(str(Path(__file__).resolve().parent.parent))  # non-pkg path hack

from neural.model_units import (  # noqa: E402  (import after path hack)
    AtomicModelUnit,
    Component,
    StaticComponent,
    ComponentIndexer,
    Featurizer,
)
from neural.pipeline import LMPipeline


# --------------------------------------------------------------------------- #
#  Token-level helper                                                         #
# --------------------------------------------------------------------------- #
class TokenPosition(ComponentIndexer):
    """Dynamic indexer: returns position(s) of interest for a prompt.

    Attributes
    ----------
    pipeline :
        The :class:`neural.pipeline.LMPipeline` supplying the tokenizer.
    """

    def __init__(self, indexer, pipeline: LMPipeline, **kwargs):
        super().__init__(indexer, **kwargs)
        self.pipeline = pipeline

    # ------------------------------------------------------------------ #
    def highlight_selected_token(self, input: dict) -> str:
        """Return *prompt* with selected token(s) wrapped in ``**bold**``.

        The method tokenizes *prompt*, calls self.index to obtain the
        positions, then re-assembles a detokenised string with the
        selected token(s) wrapped in ``**bold**``.  The rest of the
        prompt is unchanged.

        Note that whitespace handling may be approximate for tokenizers 
        that encode leading spaces as special glyphs (e.g. ``Ġ``).
        """
        ids = self.pipeline.load(input)["input_ids"][0]
        highlight = self.index(input)
        highlight = highlight if isinstance(highlight, list) else [highlight]

        return "".join(
            f"**{self.pipeline.tokenizer.decode(t)}**" if i in highlight else self.pipeline.tokenizer.decode(t)
            for i, t in enumerate(ids)
        )


# Convenience indexer
def get_last_token_index(input: dict, pipeline: LMPipeline):
    """Return a one-element list containing the *last* token index."""
    ids = list(pipeline.load(input)["input_ids"][0])
    return [len(ids) - 1]


# --------------------------------------------------------------------------- #
#  LLM-specific AtomicModelUnits                                              #
# --------------------------------------------------------------------------- #
class ResidualStream(AtomicModelUnit):
    """Residual-stream slice at *layer* for given token position(s)."""

    def __init__(
        self,
        layer: int,
        token_indices: Union[List[int], ComponentIndexer],
        *,
        featurizer: Featurizer | None = None,
        shape=None,
        feature_indices=None,
        target_output: bool = False,
    ):
        component_type = "block_output" if target_output else "block_input"
        tok_id = token_indices.id if isinstance(token_indices, ComponentIndexer) else token_indices
        uid = f"ResidualStream(Layer:{layer},Token:{tok_id})"

        unit = "pos"
        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, unit)
        else:
            component = Component(layer, component_type, token_indices, unit)

        super().__init__(
            component=component,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )


class AttentionHead(AtomicModelUnit):
    """Attention-head value stream at (*layer*, *head*) for token position(s)."""

    def __init__(
        self,
        layer: int,
        head: int,
        token_indices: Union[List[int], ComponentIndexer],
        *,
        featurizer: Featurizer | None = None,
        shape=None,
        feature_indices=None,
        target_output: bool = True,
    ):
        self.head = head
        component_type = (
            "head_attention_value_output" if target_output else "head_attention_value_input"
        )

        tok_id = token_indices.id if isinstance(token_indices, ComponentIndexer) else token_indices
        uid = f"AttentionHead(Layer:{layer},Head:{head},Token:{tok_id})"

        unit = "h.pos"

        if isinstance(token_indices, list):
            component = StaticComponent(layer, component_type, token_indices, unit)
        else:
            component = Component(layer, component_type, token_indices, unit)
        


        super().__init__(
            component=component,
            featurizer=featurizer or Featurizer(),
            feature_indices=feature_indices,
            shape=shape,
            id=uid,
        )

    # ------------------------------------------------------------------ #

    def index_component(self, input, batch=False):
        """Return indices for *input* by delegating to wrapped function."""
        if batch:
            return [[[self.head]]*len(input), [self.component.index(x) for x in input]]
        return [[[self.head]], [self.component.index(input)]]