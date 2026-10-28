"""The recorder must drain output while a pasted input stream is backpressured."""
import errno
import os
import signal
import socket
import threading
import time

import pytest

from scripts.terminalbench_pty import nonblocking, pump


@pytest.mark.parametrize('short_writes', [False, True])
def test_duplex_backpressure_preserves_bytes_and_recording(monkeypatch, short_writes):
    master, child = socket.socketpair()
    for sock in (master, child):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
    input_r, input_w = os.pipe()
    output_r, output_w = os.pipe()
    signal_r, signal_w = os.pipe()
    body = bytes(range(256))*2048
    suffix = b'final output after child input completes\n'*2048
    captured, input_recording, output_recording = [], [], []
    done, timed_out = threading.Event(), []
    original_write = os.write
    write_count = 0

    def write(fd, data):
        nonlocal write_count
        if fd in (master.fileno(), output_w):
            write_count += 1
            if write_count % 7 == 0:
                raise BlockingIOError(errno.EAGAIN, 'forced backpressure')
            return original_write(fd, data[:71])
        return original_write(fd, data)

    if short_writes:
        monkeypatch.setattr(os, 'write', write)

    def producer():
        remaining = body
        while remaining:
            remaining = remaining[original_write(input_w, remaining):]
        os.close(input_w)

    def echo():
        # The child fills its output before reading any keyboard input. The
        # original all-input-write loop deadlocks under this duplex pressure.
        child.sendall(b'initial output\n'*4096)
        received = bytearray()
        while len(received) < len(body):
            data = child.recv(1024)
            assert data
            received.extend(data)
            child.sendall(data)
        assert bytes(received) == body
        child.sendall(suffix)
        child.shutdown(socket.SHUT_WR)

    def consumer():
        while True:
            data = os.read(output_r, 2048)
            if not data:
                return
            captured.append(data)
            time.sleep(.0001)

    def watchdog():
        if not done.wait(8):
            timed_out.append(True)
            original_write(signal_w, bytes([signal.SIGTERM]))

    workers = [threading.Thread(target=f, daemon=True) for f in (producer, echo, consumer, watchdog)]
    for thread in workers:
        thread.start()
    try:
        pump(master.fileno(), [input_r, input_r], output_w, signal_r,
             lambda data: (input_recording.append(data), data)[1],
             output_recording.append, lambda sig: sig == signal.SIGTERM,
             high_water=4096)
        done.set()
        assert os.get_blocking(master.fileno())
        assert os.get_blocking(output_w) and os.get_blocking(input_r)
        os.close(output_w)
        for thread in workers:
            thread.join(1)
        assert not timed_out
        assert not any(t.is_alive() for t in workers)
        expected = b'initial output\n'*4096+body+suffix
        assert b''.join(captured) == expected
        assert b''.join(output_recording) == expected
        assert b''.join(input_recording) == body
    finally:
        done.set()
        master.close()
        child.close()
        for fd in (input_r, input_w, output_r, output_w, signal_r, signal_w):
            try:
                os.close(fd)
            except OSError:
                pass


def test_nonblocking_restores_original_flags_on_exception():
    r, w = os.pipe()
    os.set_blocking(r, False)
    try:
        with pytest.raises(RuntimeError), nonblocking([r, w, w]):
            assert not os.get_blocking(r) and not os.get_blocking(w)
            raise RuntimeError('injected')
        assert not os.get_blocking(r) and os.get_blocking(w)
    finally:
        os.close(r)
        os.close(w)


def test_invalid_buffer_limit_rejected():
    with pytest.raises(ValueError, match='Positive'):
        pump(None, [], None, None, None, None, None, high_water=0)
