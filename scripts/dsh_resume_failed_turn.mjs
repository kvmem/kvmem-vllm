// Narrow recovery endpoint: resume a failed step without appending a user Query.
// The allowlist and durable terminal sequence are supplied by the frozen audit.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';

export const name = 'kvmem-failed-turn-recovery';
export const inject = ['webServer', 'agents'];

export function resumeFailed(agent, expected) {
  if (!agent || agent.id !== expected.session_id || agent.status !== 'idle' || agent.phase?.kind !== 'idle')
    throw new Error('Recovery requires the exact idle session');
  const events = agent.session.events;
  const end = events.findLast(e => e.type === 'turn/end');
  if (!end || end.seq !== expected.terminal_seq || end.data?.reason?.error?.code !== 'TRANSPORT')
    throw new Error('Recovery terminal differs from the audited transport failure');
  if (agent.inbox.hasPending) throw new Error('Recovery cannot consume queued user input');
  const pending = new Set();
  for (const e of events) {
    if (e.type === 'tool/call') pending.add(e.data.callId);
    if (e.type === 'tool/result')
      for (const part of e.data.message?.content ?? []) if (part.toolCallId) pending.delete(part.toolCallId);
  }
  if (pending.size) throw new Error('Unsettled tools need a separate side-effect audit');
  if (typeof agent.wakeDriver !== 'function') throw new Error('Unsupported DSH loop revision');
  // DSH's existing driver starts a fresh execution boundary over the unchanged
  // transcript. followup()/steer() would insert a new user Query and are avoided.
  agent.wakeDriver();
  // This pinned DSH implementation skips a fresh, empty first step. Its async
  // preStep has yielded here; mark this as a continuation before that guard.
  // The first step's local number is already 1 and normal numbering resumes.
  // No inbox message is fabricated and no tool result is replayed for execution.
  if (agent.phase.kind !== 'running' || agent.phase.step !== 0)
    throw new Error('DSH driver did not enter the audited continuation boundary');
  agent.phase.step = 1;
  return { accepted: true, session_id: agent.id, resumed_after: end.seq, appended_user_messages: 0 };
}

export function apply(ctx) {
  const path = process.env.KVMEM_DSH_RECOVERY_PLAN;
  if (!path) throw new Error('Missing frozen DSH recovery allowlist');
  const plan = JSON.parse(readFileSync(path, 'utf8'));
  if (createHash('sha256').update(readFileSync(plan.loop_file)).digest('hex') !== plan.loop_sha256)
    throw new Error('DSH loop source changed; continuation hook requires review');
  ctx.effect(() => ctx.webServer.register({
    kind: 'exact', path: '/kvmem-resume-failed',
    async handler(req, res) {
      try {
        if (req.method !== 'POST' || !['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress))
          throw new Error('Only local recovery POST is allowed');
        const chunks=[];let bytes=0;
        for await (const chunk of req) { bytes+=chunk.length;if(bytes>4096)throw new Error('Request too large');chunks.push(chunk); }
        const body=JSON.parse(Buffer.concat(chunks).toString());
        const expected=plan.cases.find(x=>x.session_id===body.session_id && x.terminal_seq===body.terminal_seq);
        if (!expected) throw new Error('Session/boundary not authorized for recovery');
        const receipt=expected.receipt;
        if (existsSync(receipt)) { res.writeHead(200,{'content-type':'application/json'});res.end(readFileSync(receipt));return; }
        // Write-ahead marker: an ambiguous crash cannot cause duplicate wakeup.
        if (existsSync(receipt+'.started')) throw new Error('Recovery already attempted; inspect durable events');
        writeFileSync(receipt+'.started',JSON.stringify({at:Date.now(),...body}),{flag:'wx'});
        const result=resumeFailed(ctx.agents.get(body.session_id),expected);
        writeFileSync(receipt,JSON.stringify(result),{flag:'wx'});
        res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(result));
      } catch (error) { res.writeHead(409,{'content-type':'application/json'});res.end(JSON.stringify({error:String(error)})); }
    }
  }));
}
