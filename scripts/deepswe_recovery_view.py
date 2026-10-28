"""Separate original64 outcomes from user-approved five-task recovery."""
import copy
import hashlib
import json
from pathlib import Path
import time


def install_monitor(module):
    original=module.inspect
    def inspect(root,**kwargs):
        plan=module.read(root/'manifest.json',{})
        if not plan.get('recovery'):return original(root,**kwargs)
        recovery=plan['recovery'] if isinstance(plan['recovery'],dict) else {}
        campaign_name=recovery.get('campaign','recovery-1004')
        fresh=recovery.get('mode')=='fresh_restart'
        # Original seeds stay read-only. Do not run active checks against the
        # stopped original1004 controller when the new recovery is running.
        source=Path(plan['source'])
        preserved=original(source,**dict(kwargs,previous={},completion={}))
        state=module.read(root/'status.json',{});status=state.get('status')
        snapshot=dict(preserved,root=str(root),queue_status=status,active_arm=campaign_name,
                      issues=[],processes={},complete=False,all_classified_and_audited=False)
        # Preserve resource visibility failures, including failures to inspect GPU.
        snapshot['issues']=[x for x in preserved['issues'] if x.startswith(('Resource pressure:', 'GPU visibility'))]
        probe=kwargs.get('probe',module.process)
        snapshot['processes']['queue']=probe(module.read(root/'queue-process.json',{}))
        if status=='failed':snapshot['issues'].append('Recovery failed: '+str(state.get('error')))
        elif status not in ('prepared','needs_final_review','completed') and snapshot['processes']['queue']['state']!='alive':
            snapshot['issues'].append('Recovery queue not visible/alive; verify host identity')
        directory=root/campaign_name/'first16';campaign=directory.parent
        run=module.read(directory/'run.json',{});tasks=run.get('tasks',{})
        counts={s:sum(t.get('state')==s for t in tasks.values()) for s in ('completed','context_ceiling','infrastructure_error','running')}
        row=dict(status=module.read(campaign/'status.json',{}).get('status','prepared'),counts=counts,
                 expected=5,classified=counts['completed']+counts['context_ceiling'],recovery=True)
        score=module.read(campaign/'scores.json',{});audit=module.read(campaign/'execution-audit.json',{})
        row['audit_and_cleanup_passed']=(row['status']=='completed' and row['classified']==5 and
            audit.get('status')=='passed' and audit.get('classified_tasks')==5 and score.get('expected')==5 and
            len(score.get('tasks',[]))==5 and module.read(campaign/'seed-audit.json',{}).get('status')=='passed' and
            (campaign/'evidence-hashes.json').is_file() and module.clean(campaign))
        row['scored']=score.get('scored',0);row['passed']=score.get('solved',0)
        row['phase']=run.get('status','starting')
        work=module.read(directory/'server/capture/public_work.json',{})
        row['work']=work.get('completed',{});row['counter_id']=work.get('counter_id');row['engine_pid']=work.get('pid')
        row['activity_at']=module.recent_activity(directory)
        row['gc']=module.read(directory/'server/capture/archive.json',{}).get('historical_gc',{})
        previous=kwargs.get('previous') or {};now=kwargs.get('now') or time.time()
        old=previous.get('recovery',{}) if previous.get('root')==str(root) else {}
        changed=any(row.get(k)!=old.get(k) for k in ('counts','phase','work','counter_id','activity_at','gc'))
        row['last_progress_at']=now if changed else old.get('last_progress_at',now)
        if status=='running':
            for label in ('controller','server','bridge'):
                record=module.read(directory/(label+'-process.json'),{})
                if record:
                    p=probe(record);snapshot['processes']['recovery/'+label]=p
                    if p['state']!='alive' and not run.get('ended_at'):snapshot['issues'].append('Recovery '+label+' not visible/alive')
            if any(run.get(k) for k in ('error','resource_error','history_contract_error')) or counts['infrastructure_error']:
                snapshot['issues'].append('Recovery infrastructure failure; preserve all outputs and snapshots')
            if now-row['last_progress_at']>=1800:snapshot['issues'].append('Recovery no progress30min; inspect tools/GC/restore before acting')
            if row['gc'].get('policy_failures',0):snapshot['issues'].append('Recovery GC policy failures')
            for log in ('server.log','controller.log','bridge.log'):
                path=directory/log
                if path.exists() and path.stat().st_mtime<run.get('model_ready_at',0):continue
                tail,_=module.unresolved_log_tail(path,root,row['counter_id'],kwargs.get('log_resolutions') or {})
                if module.base.FATAL.search(tail):snapshot['issues'].append('Recovery inspect possible fatal error in '+log)
        if row['engine_pid'] and status=='running':
            for gpu in snapshot.get('gpu',[]):gpu['matches_active_engine']=gpu['pid']==row['engine_pid']
            if any(not g['matches_active_engine'] for g in snapshot.get('gpu',[])):
                snapshot['issues'].append('Another GPU owner visible; never stop unrelated work')
        row['mode']='fresh_restart' if fresh else 'snapshot_continuation'
        snapshot['recovery']=row
        snapshot['original_outcomes']=dict(scored=59,passed=27,infrastructure_interrupted=5,expected=64)
        if fresh:
            snapshot['selected_outcomes']=dict(expected=64,scored=59+row['scored'],passed=27+row['passed'],
                preserved_original=59,new_fresh_attempts=5,prior_attempts_preserved=True)
        if row['status']=='completed' and not row['audit_and_cleanup_passed']:
            snapshot['issues'].append('Recovery completion lacks full evidence/seed/audit/cleanup')
        completion=kwargs.get('completion') or {};report=Path(completion.get('report','/nonexistent'))
        full=(row['audit_and_cleanup_passed'] and module.clean(root) and
              module.read(root/'final-review.json',{}).get('status')=='passed' and status=='completed')
        reviewed=(completion.get('status')=='completed' and completion.get('root')==str(root) and
                  completion.get('final_review_complete') is True and report.is_file() and
                  hashlib.sha256(report.read_bytes()).hexdigest()==completion.get('report_sha256'))
        snapshot['all_classified_and_audited']=full;snapshot['complete']=full and reviewed
        return snapshot
    findings=module.findings
    def events(snapshot,previous):
        if 'recovery' not in snapshot:return findings(snapshot,previous)
        result=[x for x in snapshot['issues'] if x not in previous.get('issues',[])]
        if previous.get('issues') and not snapshot['issues']:result.append('Recovery running; original failures preserved and reported separately')
        if snapshot['recovery']['audit_and_cleanup_passed'] and not previous.get('recovery',{}).get('audit_and_cleanup_passed'):
            result.append('Five fresh restarted tasks completed; review59 preserved scores plus these five, retain all prior attempts, then write final report'
                if snapshot['recovery'].get('mode')=='fresh_restart' else
                'Five-task snapshot recovery completed; review all64 original outcomes and five recovery scores separately, then write final report')
        if snapshot['complete'] and not previous.get('complete'):result.append('Final separate original/recovery report verified; exit after delivery')
        return result
    module.inspect=inspect;module.findings=events


def install_dashboard(module):
    original_init=module.SeedDashboard.__init__;original_snapshot=module.SeedDashboard.snapshot;original_page=module.page
    def init(self,root):
        original_init(self,root)
        config=module.read_json(root/'manifest.json').get('recovery')
        self.recovery_config=config if isinstance(config,dict) else {}
        self.recovery=module.Dashboard(root/self.recovery_config.get('campaign','recovery-1004')/'first16') if config else None
    def snapshot(self):
        result=original_snapshot(self)
        if self.recovery is None:return result
        recovered=self.recovery.snapshot()
        # Keep the original denominator and all original failure rows. Only
        # the active graph/resource panel comes from the recovery cohort.
        original=copy.deepcopy(result)
        result=copy.deepcopy(recovered)
        for name in ('planned_tasks','completed_tasks','passed_tasks','context_limited_tasks','infrastructure_errors','seed_suite','run_name'):
            result[name]=original[name]
        result['tasks']=original['tasks']+[dict(r,id='恢复1004 · '+r['id'],task_id=r['id'],seed=1004,recovery=True) for r in recovered['tasks']]
        result['recovery']=dict(scored=recovered['completed_tasks'],passed=recovered['passed_tasks'],expected=5,state=recovered['state'])
        if self.recovery_config.get('mode')=='fresh_restart':
            selected={(1004,task) for task in self.recovery_config['task_ids']}
            replacements={r['id']:dict(r,id='seed 1004 · 重测 · '+r['id'],task_id=r['id'],seed=1004,fresh_restart=True)
                          for r in recovered['tasks']}
            prior=[r for r in original['tasks'] if (r['seed'],r['task_id']) in selected]
            if len(prior)!=5 or set(replacements)!=set(self.recovery_config['task_ids']):
                raise ValueError('Fresh dashboard task identity mismatch')
            result['tasks']=[replacements[r['task_id']] if (r['seed'],r['task_id']) in selected else r
                             for r in original['tasks']]
            result['prior_attempts']=prior
            result['fresh_restart']=True
            result['completed_tasks']=original['completed_tasks']+recovered['completed_tasks']
            result['passed_tasks']=original['passed_tasks']+recovered['passed_tasks']
            result['context_limited_tasks']=original['context_limited_tasks']+recovered['context_limited_tasks']
            result['infrastructure_errors']=recovered['infrastructure_errors']
            result['recovery']['mode']='fresh_restart'
            for row in result['seed_suite']['rounds']:
                if row['seed']==1004:
                    row.update(scored=11+recovered['completed_tasks'],passed=6+recovered['passed_tasks'],state=recovered['state'])
        result['original_metrics']={k:original[k] for k in ('output_tokens','requests','elapsed_seconds','e2e_output_rate','tool_counts')}
        result['state']=module.read_json(self.root/'status.json').get('status',recovered['state'])
        return result
    def page():
        value=original_page().decode()
        value=value.replace('16 个样本 × 4 个 seed · 64 次测试','64 次测试 · 历史尝试完整保留')
        marker=" if(d.seed_suite)notice="
        insert=" if(d.fresh_restart)notice=`沿用原 59 次评分，剩余 5 题按用户要求从原始题目重新测试：${d.recovery.scored}/5 已评分，${d.recovery.passed} 通过。使用新会话和原始代码，seed 1004；吞吐和资源曲线仅为本轮重测。旧中断及快照续测记录全部保留。`;\n else if(d.recovery)notice=`原始 64 次：59 次已评分、27 次通过、5 次基础设施中断，原始结果全部保留。恢复单列：${d.recovery.scored}/5 已评分，${d.recovery.passed} 通过。吞吐和资源曲线仅为这 5 题的恢复阶段；固定追加‘继续完成原任务’、冷启动重建 KV。`;\n else if(d.seed_suite)notice="
        if value.count(marker)!=1:raise ValueError('Dashboard notice anchor changed')
        return value.replace(marker,insert).encode()
    module.SeedDashboard.__init__=init;module.SeedDashboard.snapshot=snapshot;module.page=page
