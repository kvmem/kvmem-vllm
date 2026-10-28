"""Opt-in, frozen Terminus transport repair. Never imported by historical jobs.

Completion is proven only for a single submitted shell line from a known Bash
prompt. Multiline/interactive input keeps the requested observation interval.
No agent deadline, command rewrite, background process kill or answer retry.
"""
import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import time
import uuid

CONTROL = re.compile(r'(?:C-[cdz]\s*)+')
INSTRUCTIONS = '''
Local terminal executor capabilities (custom harness): standalone C-c, C-d and
C-z are keys, without a newline; repeated standalone control keys are supported.
For single-line shell commands the executor may return before duration when Bash
confirms completion. No output does NOT imply completion. Explicit idle waits and
interactive/multiline input retain duration. Executor status reports completion
and exit code or an unconfirmed/running state. A batch stops after an unconfirmed
interrupt; remaining commands are NOT sent. Decide the next action from output.
Interactive-shell curl defaults: connect timeout 15s, <1 byte/s for 120s aborts,
up to 2 retries within 300s; wget: read/connect timeout 120s, 3 tries. Explicit
command options override these defaults; scripts using their own tools are not
rewritten. There is no overall agent or CPU-command time limit. For a stuck
network/script operation use explicit bounds or send C-c and inspect the result.
'''


def controls(text):
    return re.findall(r'C-[cdz]', text) if text and CONTROL.fullmatch(text) else None


def single_line(text):
    return bool(text and text.endswith('\n') and '\n' not in text[:-1] and '\r' not in text and not controls(text))


def hook_source(path):
    # Each hook returns the incoming status so existing prompt hooks see it.
    # PROMPT_COMMAND's existing string/array entries and shell options survive.
    return '''__kb_seq=0
__kb_prompt() {
  local __kb_rc=$?
  __kb_seq=$((__kb_seq+1))
  printf '%s %s\\n' "$__kb_seq" "$__kb_rc" > PATH.tmp
  command mv PATH.tmp PATH
  printf '%s\\n' "$BASHPID" > PATH.pid
  command rm -f PATH.busy
  return "$__kb_rc"
}
if declare -p PROMPT_COMMAND 2>/dev/null | command grep -q 'declare -a'; then
  PROMPT_COMMAND=(__kb_prompt "${PROMPT_COMMAND[@]}")
else
  PROMPT_COMMAND=(__kb_prompt "${PROMPT_COMMAND-}")
fi
PS0='$(printf running > PATH.busy)'"${PS0-}"
curl() { command curl --connect-timeout 15 --speed-limit 1 --speed-time 120 --retry 2 --retry-delay 2 --retry-max-time 300 "$@"; }
wget() { command wget --timeout=120 --tries=3 "$@"; }
'''.replace('PATH', shlex.quote(path))


async def probe(session):
    prefix = shlex.quote(session._kb_path)
    target = shlex.quote(session._session_name)
    try:
        result = await session.environment.exec(command=(
            f'cat {prefix} 2>/dev/null; if test -f {prefix}.busy; then echo busy; else echo idle; fi; tmux display-message -p -t {target} '
            "'#{pane_current_command}|#{pane_dead}|#{pane_tty}'; "
            f'ps -p $(cat {prefix}.pid) --ppid $(cat {prefix}.pid) '
            '-o pid=,stat=,etime=,time=,comm= 2>/dev/null | head -c 1400'),
            user=session._user, timeout_sec=5)
    except (TimeoutError, RuntimeError) as error:
        # Harbor wraps its exec deadline in RuntimeError. This read-only probe
        # cannot prove either completion or failure of the actual command.
        # Do not swallow cancellation or unrelated environment failures.
        if isinstance(error, RuntimeError) and not re.fullmatch(
                r'Command timed out after [0-9.]+ seconds', str(error)):
            raise
        return dict(seq=None, exit_code=None, shell=False, dead=False,
                    processes='Status observation unavailable: '+str(error),
                    observation_error=type(error).__name__+': '+str(error))
    lines = (result.stdout or '').splitlines()
    seq = code = None
    if lines and re.fullmatch(r'\d+ \d+', lines[0]):
        seq, code = map(int, lines.pop(0).split())
    busy = lines.pop(0) != 'idle' if lines else True
    pane = lines.pop(0).split('|') if lines else []
    # The hook runs in the actual interactive Bash, which may be inside an
    # asciinema pseudo-terminal. tmux then names the recorder, not the shell.
    return dict(seq=seq, exit_code=code, shell=bool(seq is not None and not busy),
                dead=bool(len(pane)>1 and pane[1]=='1'), processes='\n'.join(lines))


async def prepare_recorder(session, audit):
    """Stage a private recorder adapter without changing installed packages."""
    result = await session.environment.exec(
        command='p=$(command -v asciinema) && head -n 1 "$p"', user=session._user)
    line = (result.stdout or '').strip()
    if result.return_code or not line.startswith('#!'):
        raise RuntimeError('Asciinema Python interpreter cannot be identified')
    interpreter = shlex.split(line[2:])
    if not interpreter or not interpreter[0].startswith('/'):
        raise RuntimeError('Asciinema interpreter must use an absolute path')
    source = Path(__file__).with_name('terminalbench_pty.py')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    remote = '/tmp/.kb-recorder-'+uuid.uuid4().hex+'.py'
    await session.environment.upload_file(source_path=source, target_path=remote)
    command = ' '.join(shlex.quote(x) for x in [*interpreter, remote])
    result = await session.environment.exec(command=(
        f"printf '%s  %s\\n' {digest} {remote} | sha256sum -c - && {command} --version"),
        user=session._user)
    if result.return_code:
        raise RuntimeError('Private duplex recorder qualification failed: '+str(result.stderr))
    audit(event='recorder_prepared', session=session.environment.session_id,
          source_sha256=digest, interpreter=interpreter, remote=remote,
          version=(result.stdout or '').strip(), policy='bounded-duplex-v1')
    return command


def install(audit_path, expected_sources=None, terminal_tool_bundles=None, recorder_transport=False):
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
    import inspect
    classes = (TmuxSession, Terminus2, TerminusJSONPlainParser)
    actual = {Path(inspect.getfile(c)).name: hashlib.sha256(Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in classes}
    if expected_sources is not None and actual != expected_sources:
        raise ValueError('Pinned Harbor sources changed')
    if getattr(TmuxSession, '_kb_installed', False):
        raise RuntimeError('Harness installed twice')
    TmuxSession._kb_installed = True
    audit_path = Path(audit_path)
    def audit(**value):
        with audit_path.open('a') as f:
            f.write(json.dumps(dict(at=time.time(), **value))+'\n')
    audit(event='installed', sources=actual, policy='completion-control-network-v1',
          recorder_transport=bool(recorder_transport))
    old_start, old_init = TmuxSession.start, Terminus2.__init__
    old_parse = TerminusJSONPlainParser._parse_commands
    old_tools, old_send = TmuxSession._attempt_tmux_installation, TmuxSession.send_keys
    async def tools(self):
        await old_tools(self)
        if recorder_transport and self._remote_asciinema_recording_path:
            self._kb_recorder = await prepare_recorder(self, audit)
    async def send(self, keys, *args, **kwargs):
        # Match only Harbor's own recorder startup, while start() is active.
        # Model-generated commands and terminal control bytes are untouched.
        expected=[f'asciinema rec --stdin {self._remote_asciinema_recording_path}', 'Enter']
        if getattr(self, '_kb_starting', False) and getattr(self, '_kb_recorder', None) and keys == expected:
            keys=[self._kb_recorder+' rec --stdin '+shlex.quote(str(self._remote_asciinema_recording_path)), 'Enter']
            audit(event='recorder_started', session=self.environment.session_id, command=keys[0])
        return await old_send(self, keys, *args, **kwargs)
    async def start(self):
        if terminal_tool_bundles:
            from terminalbench_tool_bundle import install_bundle
            task = Path(self.environment.environment_dir).parent.name
            if task in terminal_tool_bundles:
                await install_bundle(self, terminal_tool_bundles[task], audit)
        self._kb_starting=True
        try:
            await old_start(self)
        finally:
            self._kb_starting=False
        self._kb_path = '/tmp/.kb-terminal-'+uuid.uuid4().hex
        source = base64.b64encode(hook_source(self._kb_path).encode()).decode()
        result = await self.environment.exec(command=f'printf %s {shlex.quote(source)} | base64 -d > {self._kb_path}.sh', user=self._user)
        if result.return_code: raise RuntimeError('Cannot install terminal completion hook')
        await self.send_keys('source '+self._kb_path+'.sh\n', min_timeout_sec=.2)
        state = await probe(self)
        deadline=time.monotonic()+30
        while (state['seq'] is None or not state['shell']) and time.monotonic()<deadline:
            await asyncio.sleep(.2)
            state=await probe(self)
        self._kb_ready = bool(state['seq'] is not None and state['shell'])
        self._kb_pending = None
        self._kb_last_submitted_seq = None
        audit(event='session_ready', session=self.environment.session_id, state=state)
        if not self._kb_ready: raise RuntimeError('Shell completion hook not acknowledged')
    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self._prompt_template += INSTRUCTIONS
    def parse(self, data, warnings):
        for i, value in enumerate(data):
            if isinstance(value, dict) and isinstance(value.get('keystrokes'), str) and '\x00' in value['keystrokes']:
                return [], f'Command {i+1}: keystrokes contains a literal NUL, which cannot be sent through the terminal transport. Use textual shell escapes to represent binary bytes.'
            if isinstance(value, dict) and isinstance(value.get('duration'), (int,float)):
                if not math.isfinite(value['duration']) or value['duration'] < 0:
                    return [], f'Command {i+1}: duration must be finite and nonnegative'
        parsed, error = old_parse(self, data, warnings)
        indices = [i+1 for i,v in enumerate(data) if isinstance(v,dict) and isinstance(v.get('keystrokes'),str) and controls(v['keystrokes'])]
        warnings[:] = [w for w in warnings if not any(w.startswith(f'Command {i} should end with newline') for i in indices)]
        return parsed, error
    async def execute(self, commands, session):
        notices=[]
        for index, command in enumerate(commands):
            text=command.keystrokes; keys=controls(text)
            before=await probe(session)
            last=getattr(session, '_kb_last_submitted_seq', None)
            if (last is not None and before['seq'] is not None and
                    before['seq']>last and before['shell']):
                # A multiline command can finish after its observation window.
                # Only a newly advanced Bash prompt proves readiness; a stale
                # idle marker while Bash is reading a heredoc proves nothing.
                session._kb_ready=True
                session._kb_pending=None
            eligible=bool(single_line(text) and session._kb_ready and before['shell'])
            if keys:
                eligible=True
            if not text:
                # An explicit idle delay remains a delay; a wait on a known
                # submitted single shell line may finish when its marker arrives.
                eligible=session._kb_pending is not None
                baseline=session._kb_pending
            else:
                baseline=before['seq'] if eligible else None
                session._kb_pending=baseline
                session._kb_ready=False
                session._kb_last_submitted_seq=before['seq']
            started=time.monotonic()
            await session.send_keys(keys or text, block=False, min_timeout_sec=0)
            duration=min(command.duration_sec,60)
            if keys and 'C-c' in keys: duration=max(duration,2)
            completed=False; state=before
            while True:
                state=await probe(session)
                completed=bool(eligible and baseline is not None and state['seq'] is not None and state['seq']>baseline and state['shell'])
                remaining=duration-(time.monotonic()-started)
                if completed or state['dead'] or remaining<=0:break
                await asyncio.sleep(min(.2 if time.monotonic()-started<2 else 1.0,remaining))
            if completed:
                session._kb_pending=None;session._kb_ready=True
            elif text and not eligible and state['shell'] and state['seq'] is not None and before['seq'] is not None and state['seq']>before['seq']:
                # This is only availability for a later request, not proof for
                # early-return on a multiline batch.
                session._kb_ready=True
            elapsed=time.monotonic()-started
            notice=f'command {index+1}: '+(f'shell completion confirmed, exit={state["exit_code"]}' if completed else 'completion unconfirmed; inspect terminal/process state')+f'; observed {elapsed:.2f}s'
            notices.append(notice)
            audit(event='command', session=session.environment.session_id, original=text,
                  sent=keys or text, early_eligible=eligible, completed=completed,
                  requested_seconds=command.duration_sec, observed_seconds=elapsed,state=state)
            if keys and 'C-c' in keys and not completed:
                notices.append(f'Interrupt not confirmed: {len(commands)-index-1} following commands were NOT sent.')
                break
            if not session._kb_ready and index+1<len(commands) and commands[index+1].keystrokes and not controls(commands[index+1].keystrokes):
                notices.append(f'Shell not confirmed ready: {len(commands)-index-1} following commands were NOT sent. Inspect output before sending more input.')
                break
        output=await session.get_incremental_output()
        state=await probe(session)
        return False,self._limit_output_length(output+'\n[Executor status]\n'+'\n'.join(notices)+'\nForeground/session processes (PID state elapsed CPU-time name):\n'+state['processes'])
    TmuxSession.start=start
    TmuxSession._attempt_tmux_installation=tools
    TmuxSession.send_keys=send
    Terminus2.__init__=init
    Terminus2._execute_commands=execute
    TerminusJSONPlainParser._parse_commands=parse
    return actual
