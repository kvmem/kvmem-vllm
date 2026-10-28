import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resumeFailed } from '../scripts/dsh_resume_failed_turn.mjs';

const expected={session_id:'owned',terminal_seq:5};
function fixture() {
  const events=[{type:'tool/call',seq:3,data:{callId:'done'}},
    {type:'tool/result',seq:4,data:{message:{content:[{toolCallId:'done'}]}}},
    {type:'turn/end',seq:5,data:{reason:{kind:'error',error:{code:'TRANSPORT'}}}}];
  const agent={id:'owned',status:'idle',phase:{kind:'idle'},session:{events},inbox:{hasPending:false},wakes:0,
    wakeDriver(){this.wakes++;this.status='running';this.phase={kind:'running',step:0};}};
  return agent;
}
test('wakes only failed generation, preserving messages and completed tool events',()=>{
  const agent=fixture(),before=JSON.stringify(agent.session.events);
  assert.equal(resumeFailed(agent,expected).appended_user_messages,0);
  assert.equal(agent.wakes,1);assert.equal(JSON.stringify(agent.session.events),before);
  assert.equal(agent.phase.step,1);
  assert.throws(()=>resumeFailed(agent,expected),/idle/);
});
test('continuation bypasses only the empty first-step guard after async preStep',async()=>{
  const agent=fixture();let calls=0;
  agent.wakeDriver=function(){this.phase={kind:'running',step:0};this.status='running';
    this.done=(async()=>{const step=this.phase.step+1;await Promise.resolve();
      if(this.phase.step===0)return;calls++;this.phase.step=step;})();};
  resumeFailed(agent,expected);await agent.done;assert.equal(calls,1);
});
for(const kind of ['foreign','success','boundary','pending','tool'])test('refuses '+kind,()=>{
  const agent=fixture();
  if(kind==='foreign')agent.id='other';
  if(kind==='success')agent.session.events[2].data.reason={kind:'completed'};
  if(kind==='boundary')agent.session.events[2].seq=6;
  if(kind==='pending')agent.inbox.hasPending=true;
  if(kind==='tool')agent.session.events.splice(1,1);
  assert.throws(()=>resumeFailed(agent,expected));assert.equal(agent.wakes,0);
});
