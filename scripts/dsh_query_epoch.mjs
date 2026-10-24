// Preserve real user Query identity across the DSH -> OpenAI wire conversion.
import { createHash } from 'node:crypto';

const hash = text => createHash('sha256').update(text).digest('hex');
const textOf = content => typeof content === 'string' ? content :
  (content || []).filter(b => b.type === 'text').map(b => b.text).join('');

export function kvmemQueryEpochPayload(provider, model, options) {
  const routes = [
    ['kvmem-main-dflash2-continuation', 'kvmem-dsh-dflash2', 'http://127.0.0.1:18385/v1'],
    ['kvmem-hybrid-5050', 'kvmem-dsh-hybrid-mtp4', 'http://127.0.0.1:18386/v1'],
  ];
  if (!routes.some(([p, m, url]) => provider === p && model.id === m &&
      model.baseUrl?.replace(/\/$/, '') === url)) return undefined;
  const users = [];
  let query = null;
  for (const message of options.messages) {
    if (message.role === 'assistant') continue;
    const text = textOf(message.content);
    const results = message.content.filter(b => b.type === 'tool-result');
    if (message.role === 'system' || text.length || !results.length) {
      if (message.role === 'user' && message.source?.kind === 'user') {
        if (typeof message.id !== 'string' || !message.id) throw new Error('User Query has no stable identity');
        query = { ordinal: users.length, id: message.id };
      }
      users.push(hash(text));
    }
  }
  const digest = hash(JSON.stringify(users));
  return payload => {
    const wire = payload.messages.filter(m => m.role === 'user').map(m => hash(textOf(m.content)));
    if (hash(JSON.stringify(wire)) !== digest) throw new Error('KVMem Query provenance differs from provider user messages');
    if (payload.prompt_cache_key !== String(options.sessionId)) throw new Error('KVMem Query session identity differs');
    payload.kvmem_query_provenance = {
      schema_version: 1, session_id: String(options.sessionId), user_messages_sha256: digest,
      query_user_ordinal: query?.ordinal ?? null, query_message_id: query?.id ?? null,
    };
  };
}
