"""TCP relay from a Docker gateway address to the host's local HTTP proxy.

    python terminalbench_proxy_relay.py 172.17.0.1 17890 7890

Task containers reach the host proxy (which listens on loopback only) through
this relay when the manifest declares container_proxy. Bind it to a Docker
bridge gateway, never to a public address.
"""
import asyncio
import ipaddress
import sys


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def main(host, port, target):
    if not ipaddress.ip_address(host).is_private:
        raise SystemExit('Bind the relay to a private Docker gateway address')

    async def handle(client_reader, client_writer):
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', target)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pipe(client_reader, writer), pipe(reader, client_writer), return_exceptions=True)

    server = await asyncio.start_server(handle, host, port)
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
