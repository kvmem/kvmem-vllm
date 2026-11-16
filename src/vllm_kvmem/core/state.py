"""Generation-qualified recurrent-state checkpoints, independent of model names.

The model adapter supplies exact engine-owned state views (GDN convolution and
recurrent state, including all speculation slots). Snapshot storage is bounded.
The caller sequences capture/restore on the model stream and retains page leases.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateIdentity:
    request_id: str
    epoch: int
    history_position: int


@dataclass
class Checkpoint:
    identity: StateIdentity
    tensors: dict[str, torch.Tensor]
    event: torch.cuda.Event | None
    leases: tuple
    source_device: torch.device | None = None


class StateCheckpoints:
    def __init__(self, request_id: str, max_bytes: int, capacity: int = 2, *, offload_to_cpu: bool = False):
        if not request_id or max_bytes <= 0 or capacity <= 0:
            raise ValueError("Invalid checkpoint budget")
        self.request_id = request_id
        self.max_bytes = max_bytes
        self.capacity = capacity
        self.offload_to_cpu = offload_to_cpu
        self.epoch = 0
        self.checkpoints: dict[int, Checkpoint] = {}

    @property
    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for c in self.checkpoints.values()
                   for t in c.tensors.values())

    def capture(self, position: int, state: dict[str, torch.Tensor], leases=()) -> Checkpoint:
        if position < 0 or not state:
            raise ValueError("Checkpoint requires a nonnegative position and complete state")
        if position in self.checkpoints:
            raise ValueError("Release previous checkpoint before replacing its identity")
        required = sum(t.numel() * t.element_size() for t in state.values())
        if len(self.checkpoints) >= self.capacity or self.allocated_bytes + required > self.max_bytes:
            raise BufferError("Checkpoint budget exhausted; yield or release a completed checkpoint")
        devices = {t.device for t in state.values()}
        if len(devices) != 1:
            raise ValueError("A checkpoint must belong to one rank/device")
        device = next(iter(devices))
        if self.offload_to_cpu:
            # The model stream orders D2H before any provisional forward can
            # overwrite these pages. Retain pinned storage until the restore
            # fence completes, just as device snapshots retain their storage.
            tensors = {name: torch.empty_like(tensor, device="cpu",
                pin_memory=device.type == "cuda") for name, tensor in state.items()}
            for name, tensor in state.items():
                tensors[name].copy_(tensor, non_blocking=device.type == "cuda")
        else:
            tensors = {name: tensor.clone() for name, tensor in state.items()}
        event = None
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        checkpoint = Checkpoint(StateIdentity(self.request_id, self.epoch, position),
                                tensors, event, tuple(leases), device)
        self.checkpoints[position] = checkpoint
        return checkpoint

    def restore(self, checkpoint: Checkpoint, state: dict[str, torch.Tensor]):
        identity = checkpoint.identity
        if (identity.request_id != self.request_id or identity.epoch != self.epoch
                or self.checkpoints.get(identity.history_position) is not checkpoint):
            raise ValueError("Checkpoint belongs to another request or expired branch")
        if state.keys() != checkpoint.tensors.keys():
            raise ValueError("Incomplete model state; refusing partial restore")
        for name, dest in state.items():
            source = checkpoint.tensors[name]
            expected_device = checkpoint.source_device or source.device
            storage_valid = (source.device.type == "cpu" and
                (dest.device.type != "cuda" or source.is_pinned())) if self.offload_to_cpu else source.device == dest.device
            if (source.shape != dest.shape or source.dtype != dest.dtype or
                    dest.device != expected_device or not storage_valid):
                raise ValueError(f"State schema changed for {name}")
        device = next(iter(state.values())).device
        if checkpoint.event is not None:
            torch.cuda.current_stream(device).wait_event(checkpoint.event)
        for name, dest in state.items():
            dest.copy_(checkpoint.tensors[name], non_blocking=self.offload_to_cpu and device.type == "cuda")
        # Return a restore completion fence; callers retain this checkpoint until
        # it completes before dropping its model/page owners or reusing storage.
        event = None
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
        checkpoint.event = event
        return event

    def release(self, position: int) -> bool:
        checkpoint = self.checkpoints.get(position)
        if checkpoint is None:
            return True
        if checkpoint.event is not None and not checkpoint.event.query():
            return False
        del self.checkpoints[position]
        return True

    def new_branch(self) -> bool:
        if any(c.event is not None and not c.event.query() for c in self.checkpoints.values()):
            return False
        self.checkpoints.clear()
        self.epoch += 1
        return True
