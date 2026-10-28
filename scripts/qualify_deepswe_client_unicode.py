"""Exercise the real pinned client against an isolated local fixture server.

This checks tool execution and next-request history, not model performance.
Only Write/Read within a new temporary workspace are requested and authorized.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading


CONTENT = 'literal: \\u0000-\\u0014; actual: 中文; escaped: \\ud83d\\ude00; slash: \\\\n\n'


def run(binary, output, include_edit=False):
    output.mkdir(parents=True)
    target=output/'literal.txt';requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('count_tokens'):
                data=json.dumps({'input_tokens':100}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data);return
            index=len(requests);requests.append(request)
            if index==0:
                block=dict(type='tool_use',id='tool_write',name='Write',input=dict(file_path=str(target),content=CONTENT))
            elif index==1:
                block=dict(type='tool_use',id='tool_read',name='Read',input=dict(file_path=str(target)))
            elif include_edit and index==2:
                block=dict(type='tool_use',id='tool_edit',name='Edit',input=dict(file_path=str(target),
                    old_string=r'\u0000-\u0014',new_string=r'\u0041-\u0042'))
            elif include_edit and index==3:
                block=dict(type='tool_use',id='tool_read_after',name='Read',input=dict(file_path=str(target)))
            else:block=dict(type='text',text='Fixture round trip complete.')
            is_tool=block['type']=='tool_use'
            reason='tool_use' if is_tool else 'end_turn'
            message=dict(id='msg_fixture_'+str(index),type='message',role='assistant',model='local-fixture',
                content=[],stop_reason=None,stop_sequence=None,usage=dict(input_tokens=100,output_tokens=0))
            start_block={**block,**({'input':{}} if is_tool else {'text':''})}
            delta=dict(type='input_json_delta',partial_json=json.dumps(block['input'])) if is_tool else dict(type='text_delta',text=block['text'])
            events=[('message_start',dict(type='message_start',message=message)),
                    ('content_block_start',dict(type='content_block_start',index=0,content_block=start_block)),
                    ('content_block_delta',dict(type='content_block_delta',index=0,delta=delta)),
                    ('content_block_stop',dict(type='content_block_stop',index=0)),
                    ('message_delta',dict(type='message_delta',delta=dict(stop_reason=reason,stop_sequence=None),usage=dict(output_tokens=20))),
                    ('message_stop',dict(type='message_stop'))]
            data=''.join('event: '+event+'\ndata: '+json.dumps(value)+'\n\n' for event,value in events).encode()
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={name:os.environ[name] for name in ['PATH','LD_LIBRARY_PATH','LANG'] if name in os.environ}
    env.update(CLAUDE_CONFIG_DIR=str(output/'config'),ANTHROPIC_BASE_URL=f'http://127.0.0.1:{server.server_port}',
        ANTHROPIC_API_KEY='local-fixture-only',ANTHROPIC_AUTH_TOKEN='local-fixture-only',
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',DISABLE_COMPACT='1')
    try:
        with (output/'client.log').open('x') as log:
            result=subprocess.run([str(binary),'-p','--verbose','--output-format','stream-json',
                '--permission-mode','acceptEdits','--allowedTools','Write,Read,Edit' if include_edit else 'Write,Read',
                '--setting-sources','project,local','--settings','{}','--model','local-fixture','--',
                f'Write the exact supplied fixture content to {target}, then read that file.'
                + (' Edit the supplied literal substring and read it again.' if include_edit else '')
                + ' This is an authorized isolated tool test.'],
                cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    finally:
        server.shutdown();server.server_close();thread.join()
        (output/'requests.json').write_text(json.dumps(requests,ensure_ascii=True,indent=2)+'\n')
    history=[block for request in requests[1:] for message in request.get('messages',[]) if message.get('role')=='assistant'
             for block in message.get('content',[]) if isinstance(block,dict) and block.get('id')=='tool_write']
    report=dict(client_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),exit=result.returncode,
        requests=len(requests),file_exists=target.exists(),
        exact_file=target.exists() and target.read_text()==(CONTENT.replace(r'\u0000-\u0014',r'\u0041-\u0042') if include_edit else CONTENT),
        exact_history=bool(history) and all(v['input'].get('content')==CONTENT for v in history),
        scope='Actual client Write/Read plus serialized next-request history. Local fixture model, not GPU or throughput evidence.')
    if include_edit:
        edits=[block for request in requests[3:] for message in request.get('messages',[]) if message.get('role')=='assistant'
               for block in message.get('content',[]) if isinstance(block,dict) and block.get('id')=='tool_edit']
        report.update(exact_edit_history=bool(edits) and all(v['input'].get('old_string')==r'\u0000-\u0014'
            and v['input'].get('new_string')==r'\u0041-\u0042' and v['input'].get('replace_all',False) is False for v in edits),
            scope='Actual client Write/Read/Edit/Read and next-request history. Local fixture model, no GPU throughput claim.')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-edit',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args.binary.resolve(),args.output.resolve(),args.include_edit)))
