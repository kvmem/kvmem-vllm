import errno
import socket

import pytest

from scripts.run_agent_handoff_queue import require_free_port


def test_closed_listener_time_wait_does_not_block_next_arm():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
        server.listen(1)
        with socket.create_connection(('127.0.0.1', port)) as client:
            connection, _ = server.accept()
            with connection:
                # Server actively closes, leaving server-port TIME_WAIT.
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b''
    with socket.socket() as old_probe:
        with pytest.raises(OSError) as error:
            old_probe.bind(('127.0.0.1', port))
        assert error.value.errno == errno.EADDRINUSE
    require_free_port(port)


@pytest.mark.parametrize('address', ['127.0.0.1', '0.0.0.0'])
def test_live_listener_is_never_displaced(address):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((address, 0))
        port = server.getsockname()[1]
        server.listen(1)
        with pytest.raises(OSError) as error:
            require_free_port(port)
        assert error.value.errno == errno.EADDRINUSE
        with socket.create_connection(('127.0.0.1', port)) as client:
            connection, _ = server.accept()
            with connection:
                connection.sendall(b'alive')
                assert client.recv(5) == b'alive'
