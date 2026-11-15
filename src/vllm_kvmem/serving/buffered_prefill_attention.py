"""Opt-in immutable native state for the unchanged cumulative attention rule.

Each staged history shares its previous immutable state. A successful commit
returns a new state; rejected or failed packets cannot mutate a prior snapshot.
The existing dictionary interface is materialized only for scoring/checkpoints.
"""
import math
import os

from .prefill_attention import PrefillHistory
from ..core import settings as _settings


def require_batch_accept():
    from .. import _prefill_state_native
    if not callable(getattr(_prefill_state_native,'committed_batch',None)):
        raise ValueError('Batched layer acceptance requires the rebuilt prefill state extension')
    return _prefill_state_native.committed_batch


def accept_buffered_batch(histories,end,count,accepted,rows,starts,logical,local_begin,masses):
    """Update staged histories together; publication remains the caller's transaction."""
    # DeferredFoldHistory (deferred_fold.py) forces its queued folds on access.
    if any(type(history) is not BufferedPrefillHistory and
           not getattr(type(history), 'deferred_fold', False) for history in histories):
        raise ValueError('Batched layer acceptance requires native buffered histories')
    # Preserve skipped/mixed sampling boundaries and malformed outer geometry's
    # scalar indexing/error order. Normal downloaded packets always batch.
    if (not histories or accepted!=count or not rows or
            any(end<=history.end for history in histories) or
            len(logical)!=len(histories) or getattr(masses,'ndim',None)!=3 or
            masses.shape[0]!=len(histories)):
        return [history.accept(end,count,accepted,rows,starts,logical[layer],local_begin,masses[layer])
                for layer,history in enumerate(histories)]
    if hasattr(masses,'numpy'):
        buffer=masses.numpy()
    else:
        import numpy as np
        buffer=np.asarray(masses)
        if buffer.dtype.kind!='f':buffer=buffer.astype('float64')
    decays=[math.exp2(-(end-history.end)/history.options.prefill_attention_half_life)
            for history in histories]
    for history in histories:
        history.last_commit_backend=None
    # history_shift maps a row's window position to its history position (lift distances).
    updated=require_batch_accept()([history._buffered_state for history in histories],
        decays,rows,starts,logical,local_begin,buffer,end-count-local_begin)
    for history,state in zip(histories,updated):
        history._buffered_state=state
        history.end=end;history.samples+=len(rows);history.chunks+=1
        history.successful_buffered_commits+=1
        history.successful_buffered_batch_commits+=1
        history.last_commit_backend='native_buffered'
    return [True]*len(histories)


class BufferedPrefillHistory(PrefillHistory):
    storage_backend='native_buffered'

    @property
    def values(self):
        return self._buffered_state.values()

    @values.setter
    def values(self, values):
        from .._prefill_state_native import State
        # A lift state keeps its expected attention and distance profile when GC prunes
        # keys (strategy_pressure_scorer=lift); it refuses to gain keys this way.
        current=getattr(self,'_buffered_state',None)
        self._buffered_state=(current.with_values(values) if getattr(current,'lift',False)
                              else State.from_values(values))

    def __init__(self,options):
        super().__init__(options)
        if not self.native_commit:
            raise ValueError('Buffered history requires the qualified native commit policy')
        self.successful_buffered_commits=0
        self.successful_buffered_batch_commits=0
        self.successful_buffered_restores=0
        self.native_score_export=_settings.get('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT')=='1'
        if self.native_score_export and not hasattr(self._buffered_state,'scores'):
            raise ValueError('Native score export requires the rebuilt prefill state extension')

    def page_count(self):
        state = self._buffered_state
        return state.size() if hasattr(state, 'size') else len(state.values())

    def scores(self):
        # Keep the qualified dictionary path by default, even in a new build.
        # The separate execution candidate must be explicitly enabled.
        return self._buffered_state.scores() if self.native_score_export else super().scores()

    def restore(self,snapshot,position):
        super().restore(snapshot,position)
        self.successful_buffered_restores+=1

    def accept(self,end,count,accepted,rows,starts,history,local_begin,masses):
        self.last_commit_backend=None
        if accepted!=count or end<=self.end or not rows:return False
        if hasattr(masses,'numpy'):
            buffer=masses.numpy()
        else:
            import numpy as np
            buffer=np.asarray(masses)
            if buffer.dtype.kind!='f':buffer=buffer.astype('float64')
        decay=math.exp2(-(end-self.end)/self.options.prefill_attention_half_life)
        updated=self._buffered_state.committed(decay,rows,starts,history,local_begin,buffer,end-count-local_begin)
        self._buffered_state=updated
        self.end=end;self.samples+=len(rows);self.chunks+=1
        self.successful_buffered_commits+=1
        self.last_commit_backend='native_buffered'
        return True
