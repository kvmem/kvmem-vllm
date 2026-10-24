import test from 'node:test';
import assert from 'node:assert/strict';
import { kvmemQueryEpochPayload } from '../scripts/dsh_query_epoch.mjs';

const model={id:'kvmem-dsh-dflash2',baseUrl:'http://127.0.0.1:18385/v1'};
const message=(id,kind,text)=>({id,role:'user',source:{kind},content:[{type:'text',text}]});
const options=()=>({sessionId:'s',messages:[message('u1','user','任务 中文'),
  {id:'tool',role:'user',source:{kind:'tool'},content:[{type:'tool-result',content:[]}]},
  message('p','plugin','notice') ]});
const wire=()=>({prompt_cache_key:'s',messages:[{role:'system',content:'system'},
  {role:'user',content:'任务 中文'},{role:'tool',content:'result'},
  {role:'user',content:'notice'}]});

test('plugin and tool messages do not replace the manual Query',()=>{
  const body=wire(),before=JSON.stringify(body.messages);
  kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,options())(body);
  assert.equal(body.kvmem_query_provenance.query_user_ordinal,0);
  assert.equal(body.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(JSON.stringify(body.messages),before);
});

test('identical text sent manually again starts a distinct Query',()=>{
  const o=options();o.messages.push(message('u2','user','任务 中文'));
  const b=wire();b.messages.push({role:'user',content:'任务 中文'});
  kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,o)(b);
  assert.equal(b.kvmem_query_provenance.query_user_ordinal,2);
  assert.equal(b.kvmem_query_provenance.query_message_id,'u2');
});

test('wire mismatch and wrong session fail before attaching proof',()=>{
  const callback=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,options());
  const a=wire();a.messages[1].content='different';assert.throws(()=>callback(a));
  const b=wire();b.prompt_cache_key='other';assert.throws(()=>callback(b));
  assert.equal(a.kvmem_query_provenance,undefined);assert.equal(b.kvmem_query_provenance,undefined);
});

test('scope is the owned local route and each callback owns its source identity',()=>{
  assert.equal(kvmemQueryEpochPayload('other',model,options()),undefined);
  assert.equal(kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',{...model,baseUrl:'https://example.com'},options()),undefined);
  const a=options(),b=options();b.messages[0].id='other-id';
  const one=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,a);
  const two=kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',model,b);
  const x=wire(),y=wire();two(y);one(x);
  assert.equal(x.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(y.kvmem_query_provenance.query_message_id,'other-id');
});

test('isolated hybrid route retains manual Query identity with strict endpoint scoping',()=>{
  const hybrid={id:'kvmem-dsh-hybrid-mtp4',baseUrl:'http://127.0.0.1:18386/v1/'};
  const b=wire();
  kvmemQueryEpochPayload('kvmem-hybrid-5050',hybrid,options())(b);
  assert.equal(b.kvmem_query_provenance.query_message_id,'u1');
  assert.equal(kvmemQueryEpochPayload('kvmem-hybrid-5050',model,options()),undefined);
  assert.equal(kvmemQueryEpochPayload('kvmem-main-dflash2-continuation',hybrid,options()),undefined);
});
