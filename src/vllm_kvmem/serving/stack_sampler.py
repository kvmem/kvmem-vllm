"""Opt-in in-process Python stack sampler for the engine thread (diagnostic).

KVMEM_STACK_PROFILE_DIR=<dir> starts a daemon thread that waits for a file
``<dir>/go`` (its content: seconds to sample, default 15), then samples the
thread that built the model runner every millisecond and writes collapsed
stacks (``frame;frame;... count``, outermost first) to ``<dir>/stacks.txt``.
Each sample holds the GIL briefly (about 3% overhead at 1 kHz). Frames are
``function (file.py:line)`` of the call site, so time the engine thread spends
inside C calls (CUDA syncs, torch ops) is attributed to the Python line that
made them. Built for processes py-spy cannot attach to.
"""
import os
import sys
import threading
import time
from collections import Counter


def _stack(frame):
    out = []
    while frame is not None:
        code = frame.f_code
        out.append(f'{code.co_name} ({os.path.basename(code.co_filename)}:{frame.f_lineno})')
        frame = frame.f_back
    out.reverse()
    return ';'.join(out)


def _run(directory, target):
    trigger = os.path.join(directory, 'go')
    while not os.path.exists(trigger):
        time.sleep(1.0)
    try:
        seconds = float(open(trigger).read().strip() or 15)
    except (OSError, ValueError):
        seconds = 15.0
    counts, others, samples = Counter(), Counter(), 0
    names = {}
    own = threading.get_ident()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frames = sys._current_frames()
        frame = frames.get(target)
        if frame is not None:
            counts[_stack(frame)] += 1
            samples += 1
        # Other threads: their innermost three frames (a thread waiting in C
        # shows the Python line that made the call: lock waits, sleeps, I/O).
        for ident, other in frames.items():
            if ident in (target, own):
                continue
            name = names.get(ident)
            if name is None:
                name = names[ident] = next((t.name for t in threading.enumerate() if t.ident == ident), str(ident))
            top, depth = [], 0
            while other is not None and depth < 3:
                code = other.f_code
                top.append(f'{code.co_name} ({os.path.basename(code.co_filename)}:{other.f_lineno})')
                other, depth = other.f_back, depth + 1
            others[name + ';' + ';'.join(top)] += 1
        del frame, frames
        time.sleep(0.001)
    path = os.path.join(directory, 'stacks.txt')
    with open(path + '.tmp', 'w') as handle:
        for stack, count in counts.most_common():
            handle.write(f'{stack} {count}\n')
    os.replace(path + '.tmp', path)
    with open(os.path.join(directory, 'threads.txt'), 'w') as handle:
        for stack, count in others.most_common():
            handle.write(f'{stack} {count}\n')
    with open(os.path.join(directory, 'done'), 'w') as handle:
        handle.write(f'{samples} samples over {seconds:.1f} s\n')


def install(directory):
    """Start the sampler for the calling thread (the engine thread)."""
    os.makedirs(directory, exist_ok=True)
    thread = threading.Thread(target=_run, args=(directory, threading.get_ident()),
                              name='kvmem-stack-sampler', daemon=True)
    thread.start()
    return thread
