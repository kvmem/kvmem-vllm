"""Read-only browser smoke checks for completed isolated DSH workspaces.

Writes screenshots/evidence outside generated projects. No repair prompts are
injected. This checks loading and basic interaction, not complete game correctness.
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time

from playwright.sync_api import sync_playwright


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass


def check(browser,root,row):
    output=root/'cases'/row['id']/'browser';output.mkdir(exist_ok=False)
    workspace=Path(row['workspace'])
    if not (workspace/'index.html').exists():
        save(output/'result.json',dict(status='failed',reason='Missing index.html'));return
    server=ThreadingHTTPServer(('127.0.0.1',row['preview_port']),
        functools.partial(QuietHandler,directory=str(workspace)))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    ctx=browser.new_context(viewport={'width':1280,'height':800})
    page=ctx.new_page();errors=[];console=[];network=[];actions=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('console',lambda m:console.append(dict(type=m.type,text=m.text)) if m.type=='error' else None)
    page.on('requestfailed',lambda r:network.append(dict(url=r.url,error=r.failure)))
    report=dict(task=row['task'],page_errors=errors,console_errors=console,request_failures=network,actions=actions,
        scope='Page load, start if available, basic controls, resize and reload; does not establish full feature correctness.')
    try:
        response=page.goto(f'http://127.0.0.1:{row["preview_port"]}/index.html',wait_until='load',timeout=30000)
        page.wait_for_timeout(1000)
        report.update(http_status=response.status,title=page.title(),body_text=page.locator('body').inner_text()[:12000],
            buttons=page.locator('button').all_text_contents(),canvas_count=page.locator('canvas').count(),
            input_count=page.locator('input,textarea,select').count())
        page.screenshot(path=str(output/'initial.png'))
        if row['task'] in ('fps','tower','snake','pomodoro'):
            candidates=page.get_by_role('button',name=re.compile('^(开始|新游戏|进入游戏|Start|Play|开始游戏|开始战斗|开始专注)',re.I))
            for i in range(candidates.count()):
                if candidates.nth(i).is_visible():
                    actions.append(dict(action='click',label=candidates.nth(i).inner_text()))
                    candidates.nth(i).click(timeout=3000);break
            if row['task'] in ('fps','snake'):
                for key in (['w','d','r'] if row['task']=='fps' else ['ArrowRight','ArrowDown','ArrowLeft']):
                    page.keyboard.down(key);page.wait_for_timeout(250);page.keyboard.up(key)
                    actions.append(dict(action='keyboard',key=key))
                page.keyboard.press('Escape')
        page.wait_for_timeout(700);page.screenshot(path=str(output/'interaction.png'))
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
        page.screenshot(path=str(output/'mobile.png'))
        report['mobile_horizontal_overflow']=page.evaluate('document.documentElement.scrollWidth > innerWidth + 2')
        page.reload(wait_until='load');page.wait_for_timeout(300)
        report['reload_title']=page.title()
        report['status']='passed_basic_smoke' if not errors and response.status==200 else 'failed'
    except Exception as error:
        report.update(status='failed',exception=repr(error))
    finally:
        save(output/'result.json',report);ctx.close();server.shutdown();server.server_close();worker.join(3)


def run(root,wait):
    plan=json.loads((root/'task-plan.json').read_text());deadline=time.time()+24*3600
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/home/chaidi/.cache/puppeteer/chrome/linux-148.0.7778.97/chrome-linux64/chrome',
            headless=True,args=['--no-sandbox','--disable-gpu','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
        while True:
            for row in plan['cases']:
                case=root/'cases'/row['id'];state=case/'status.json'
                if (case/'browser/result.json').exists():continue
                if state.exists() and json.loads(state.read_text()).get('status')=='completed':check(browser,root,row)
            reports={row['id']:json.loads((root/'cases'/row['id']/'browser/result.json').read_text())
                for row in plan['cases'] if (root/'cases'/row['id']/'browser/result.json').exists()}
            save(root/'browser-summary.json',dict(checked=len(reports),expected=24,reports=reports,at=time.time()))
            state=json.loads((root/'status.json').read_text()).get('status')
            if not wait or state=='needs_attention' or len(reports)==24 or time.time()>deadline:break
            time.sleep(15)
        browser.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args();run(args.root,args.wait)
