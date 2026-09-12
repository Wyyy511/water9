from __future__ import annotations
import json, math, os, tempfile, uuid, sys, shutil, re
from pathlib import Path

# Railway/Railpack may start Uvicorn with a working directory that is not
# automatically added to Python's import path. Force the repository root
# onto sys.path before importing local packages such as tools/, agent/, core/.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# V8.7 deployment hardening: server.py repairs flat uploads by itself.
# There is NO dependency on bootstrap_layout.py.
_LAYOUT_GROUPS={
"tools":["data_tool.py","query_adapter.py","baseline_tool.py","scenario_tool.py","export_tool.py","qa_tool.py","field_governance.py","material_tool.py"],
"agent":["router.py","deepseek_client.py","llm_adapter.py","risk_brief.py"],
"core":["logging_utils.py","paths.py"],
"engines":["baseline_api.py","baseline_calculator.py","baseline_water_risk_v2.py","scenario_engine.py"],
"static":["index.html","app.js","styles.css","Water_Risk_Input_Template.xlsx"],
"data":["data_registry.json","exposure_weights.csv","extreme_drought.csv","field_dictionary.csv","field_gap_register.csv","future_ws_sv.csv","hazard_baseline.csv","material_params.csv","monthly_ws.csv","scenario_params.json"],
}
_GOLDEN=["example_1_complete_input.json","example_1_complete_output.json","example_2_proxy_input.json","example_2_proxy_output.json","example_3_partial_input.json","example_3_partial_output.json","example_4_insufficient_input.json","example_4_insufficient_output.json","example_5_conflict_input.json","example_5_conflict_output.json","example_6_error_input.json","example_6_error_output.json"]
def _ensure_runtime_layout():
    for folder,names in _LAYOUT_GROUPS.items():
        dest=ROOT/folder;dest.mkdir(parents=True,exist_ok=True)
        if folder in {"tools","agent","core","engines"}:(dest/"__init__.py").touch(exist_ok=True)
        for name in names:
            target=dest/name
            if not target.exists():
                source=ROOT/name
                if not source.exists():raise RuntimeError(f"部署文件缺失：{name}")
                shutil.copy2(source,target)
    golden=ROOT/"qa"/"golden";golden.mkdir(parents=True,exist_ok=True)
    for name in _GOLDEN:
        target=golden/name
        if not target.exists():
            source=ROOT/name
            if not source.exists():raise RuntimeError(f"部署文件缺失：{name}")
            shutil.copy2(source,target)
    for folder in ["logs","outputs","sample_inputs"]:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    return {"mode":"server-self-healing","reconstructed":True}
BOOTSTRAP_INFO=_ensure_runtime_layout()
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(ROOT/'.env')

from tools.data_tool import read_user_file, suggest_mapping, apply_mapping, validate_normalized, load_demo_procurement
from tools.query_adapter import resolve_location, lookup_water_risk, lookup_material_parameters, supported_locations
from tools.baseline_tool import calculate_baseline
from tools.scenario_tool import run_scenario
from tools.export_tool import export_risk_report
from tools.qa_tool import run_qa
from tools.field_governance import contract_summary, evaluate_user_records, user_requirements_text, deepseek_context
from agent.router import route_intent
from agent.deepseek_client import configured as deepseek_configured, model_name as deepseek_model, api_key_source as deepseek_key_source, base_url as deepseek_base_url, classify_intent, extract_supply_chain as deepseek_extract_supply_chain, general_guidance, analyze_tool_result, connection_test as deepseek_connection_test
from agent.risk_brief import baseline_brief, scenario_brief, data_audit_brief
from core.logging_utils import log_event, read_logs, new_run_id, now_iso, LOG_FILE

app=FastAPI(title='WaterPulse AI Agent API', version='9.2')

@app.middleware('http')
async def no_cache_ui(request, call_next):
    response=await call_next(request)
    if request.url.path=='/' or request.url.path.endswith(('.html','.js','.css')):
        response.headers['Cache-Control']='no-store, no-cache, must-revalidate, max-age=0'
    return response


def clean_scalar(v: Any):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): 
        x=float(v); return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    return v


def jsonable(x: Any):
    if isinstance(x, pd.DataFrame):
        return [{str(k):jsonable(v) for k,v in row.items()} for row in x.to_dict(orient='records')]
    if isinstance(x, pd.Series): return {str(k):jsonable(v) for k,v in x.to_dict().items()}
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,set)): return [jsonable(v) for v in x]
    return clean_scalar(x)


def restore_result(obj: dict | None) -> dict | None:
    if not obj: return obj
    out=dict(obj)
    d=out.get('data')
    if isinstance(d,dict):
        d=dict(d)
        if isinstance(d.get('summary'),list): d['summary']=pd.DataFrame(d['summary'])
        if isinstance(d.get('nodes'),list): d['nodes']=pd.DataFrame(d['nodes'])
        if isinstance(d.get('node_table'),list): d['node_table']=pd.DataFrame(d['node_table'])
        if isinstance(d.get('replacement_table'),list): d['replacement_table']=pd.DataFrame(d['replacement_table'])
        out['data']=d
    return out


class ConfirmRequest(BaseModel):
    records: list[dict]
    mapping: list[dict]
    weight_mode: str='0-1'
    human_confirmed: bool=False

class BaselineRequest(BaseModel):
    records: list[dict]

class ScenarioRequest(BaseModel):
    baseline: dict
    scenario_type: str
    material: str
    year: int=2050
    path: str='BAU'
    failure_fraction: float=1.0
    inventory: float=0.10

class ChatRequest(BaseModel):
    message: str
    validation: dict | None=None
    baseline: dict | None=None
    scenario: dict | None=None

class ExportRequest(BaseModel):
    baseline: dict
    scenario: dict | None=None
    validation: dict | None=None
    snapshot_id: str=''


@app.get('/api/health')
def health():
    return {'status':'ok','app':'WaterPulse AI Agent','version':'9.2','deployment_layout':'server-self-healing','deepseek_configured':deepseek_configured(),'deepseek_model':deepseek_model(),'field_contract':contract_summary(),'timestamp':now_iso()}


@app.get('/api/deepseek/test')
def deepseek_test():
    return deepseek_connection_test()

@app.get('/api/qa')
def qa():
    """QA 黄金样例回归：6 组固定输入跑确定性引擎，与固定期望输出逐项比对。"""
    r=run_qa()
    log_event({'event':'qa_golden','status':r.get('status'),'passed':r.get('n_pass'),'total':r.get('n_total')})
    return r

@app.get('/api/demo/{material}')
def demo(material: str):
    if material not in ['甘蔗','甜菜','大豆']:
        raise HTTPException(404,'未知材料')
    df=load_demo_procurement(material)
    mapping=[
        {'system_field':'enterprise','source_column':'enterprise','required':False},
        {'system_field':'material','source_column':'material','required':True},
        {'system_field':'node_id','source_column':'node_id','required':False},
        {'system_field':'node_name','source_column':'node_name','required':False},
        {'system_field':'purchase_weight','source_column':'purchase_weight','required':True},
        {'system_field':'year','source_column':'year','required':False},
    ]
    return {'status':'success','records':jsonable(df),'mapping':mapping,
            'warning':'示范采购权重为 A 类研究假设，不代表真实企业采购。'}

@app.post('/api/upload')
async def upload(file: UploadFile=File(...)):
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in ['.csv','.xlsx','.xls','.docx','.pdf']:
        raise HTTPException(400,'仅支持 CSV/XLSX/XLS/DOCX/PDF')
    payload=await file.read()
    tmp=ROOT/'outputs'/f'upload_{uuid.uuid4().hex[:10]}{suffix}'
    tmp.write_bytes(payload)
    try:
        res=read_user_file(str(tmp))
        df=res.get('data') if isinstance(res.get('data'),pd.DataFrame) else pd.DataFrame()
        if res.get('kind')=='table':
            mapping=suggest_mapping(list(df.columns))
            msg=f'已读取表格 {len(df)} 行 × {len(df.columns)} 列。请确认字段映射后再进入计算。'
        else:
            mapping=pd.DataFrame([
                {'system_field':'enterprise','source_column':'enterprise','required':False},
                {'system_field':'material','source_column':'material','required':True},
                {'system_field':'node_id','source_column':'node_id','required':False},
                {'system_field':'node_name','source_column':'node_name','required':False},
                {'system_field':'purchase_weight','source_column':'purchase_weight','required':True},
                {'system_field':'year','source_column':'year','required':False},
            ])
            msg='已生成候选供应链记录。文档抽取只是候选数据，关键采购字段必须人工确认。'
        log_event({'event':'upload_parse','filename':file.filename,'kind':res.get('kind'),'rows':len(df)})
        return {'status':'success','kind':res.get('kind'),'records':jsonable(df),'mapping':jsonable(mapping),'message':msg,'text_preview':(res.get('text') or '')[:12000]}
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass

@app.post('/api/confirm')
def confirm(req: ConfirmRequest):
    if not req.human_confirmed:
        return {'status':'insufficient','message':'请先人工确认关键字段映射与采购口径。'}
    raw=pd.DataFrame(req.records)
    mp=pd.DataFrame(req.mapping)
    normalized=apply_mapping(raw,mp,req.weight_mode)
    validation=validate_normalized(normalized)
    snapshot_id='snap_'+new_run_id()
    log_event({'event':'input_confirmed','snapshot_id':snapshot_id,'status':validation.get('status'),'rows':len(normalized),'validation':validation})
    return {'status':validation.get('status'),'snapshot_id':snapshot_id,'records':jsonable(normalized),'validation':jsonable(validation),'audit_brief':data_audit_brief(validation)}

@app.post('/api/resolve-input')
def resolve_input(req: BaselineRequest):
    """Enrich confirmed enterprise input with the currently registered local B/C data snapshot.
    This endpoint is deliberately adapter-based so the formal query kernel can replace it later.
    """
    out=[]; warnings=[]; overall='success'
    for row in req.records:
        r=dict(row)
        node_id=str(r.get('node_id') or '').strip()
        if not node_id and r.get('node_name'):
            loc=resolve_location(str(r.get('node_name')), str(r.get('material') or ''))
            if loc.get('data'):
                node_id=loc['data']['node_id']; r['node_id']=node_id; r['node_name']=loc['data']['node_name']
            if loc.get('status')!='success': overall='partial'
            warnings.extend(loc.get('warnings',[]))
        wr=lookup_water_risk(node_id)
        mp=lookup_material_parameters(str(r.get('material') or ''))
        if wr.get('status')=='insufficient' or mp.get('status')=='insufficient': overall='insufficient'
        elif wr.get('status')!='success' or mp.get('status')!='success': overall='partial' if overall!='insufficient' else overall
        r['hazard']=wr.get('data')
        r['material_parameters']=mp.get('data')
        warnings.extend(wr.get('warnings',[])); warnings.extend(mp.get('warnings',[]))
        out.append(r)
    return {'status':overall,'resolved_input':out,'warnings':list(dict.fromkeys(warnings)),'data_release_id':'integrated-mvp-2026-09-06'}

@app.post('/api/baseline')
def baseline(req: BaselineRequest):
    df=pd.DataFrame(req.records)
    result=calculate_baseline(df)
    log_event({'event':'baseline','status':result.get('status'),'model_version':result.get('model_version'),'warnings':result.get('warnings'),'gaps':result.get('data_gaps')})
    j=jsonable(result)
    j['brief']=baseline_brief(result)
    return j

@app.post('/api/scenario')
def scenario(req: ScenarioRequest):
    b=restore_result(req.baseline)
    result=run_scenario(b,req.scenario_type,req.material,req.year,req.path,req.failure_fraction,req.inventory)
    log_event({'event':'scenario','status':result.get('status'),'scenario_type':req.scenario_type,'material':req.material,'warnings':result.get('warnings'),'gaps':result.get('data_gaps')})
    j=jsonable(result)
    j['brief']=scenario_brief(result)
    return j

@app.post('/api/chat')
def chat(req: ChatRequest):
    route=route_intent(req.message)
    b=restore_result(req.baseline)
    s=restore_result(req.scenario)
    if route['intent']=='data_audit':
        answer=data_audit_brief(req.validation or {}); tools=['data_validation']
    elif route['intent']=='baseline':
        if b and b.get('data') is not None:
            answer=baseline_brief(b); tools=['lookup_water_risk','lookup_material_parameters','calculate_baseline']
        else:
            answer='该问题属于 Baseline 诊断，但当前尚无有效 Baseline。请先完成数据确认与计算。'; tools=[]
    elif route['intent']=='scenario':
        if s and s.get('data') is not None:
            answer=scenario_brief(s); tools=['calculate_baseline','run_scenario','generate_risk_brief']
        else:
            answer='该问题属于情景压力测试。请先完成 Baseline 并运行对应 Scenario；Agent 不会在没有工具结果时生成情景数字。'; tools=[]
    else:
        if s and s.get('data') is not None:
            answer=scenario_brief(s)+'\n\n'+(baseline_brief(b) if b else '') ; tools=['generate_risk_brief']
        elif b and b.get('data') is not None:
            answer=baseline_brief(b); tools=['generate_risk_brief']
        else:
            answer='你可以先上传相关资料。我会在信息足够后告诉你主要风险来自哪里、哪些供应地更值得关注，以及下一步可以怎么做；在数据不足时不会凭空给出数字。'; tools=[]
    log_event({'event':'agent_chat','user_query':req.message,'intent':route['intent'],'scenario_type':route.get('scenario_type'),'tool_calls':tools})
    return {'status':'success','intent':route['intent'],'scenario_type':route.get('scenario_type'),'confidence':route.get('confidence'),'tool_calls':tools,'answer':answer}

@app.post('/api/export')
def export(req: ExportRequest):
    b=restore_result(req.baseline); s=restore_result(req.scenario)
    if not b or b.get('data') is None:
        raise HTTPException(400,'请先运行 Baseline')
    path=export_risk_report(b,s,req.validation,req.snapshot_id)
    log_event({'event':'export_report','path':path,'snapshot_id':req.snapshot_id})
    return FileResponse(path,filename=Path(path).name,media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

@app.get('/api/logs')
def logs():
    return {'status':'success','logs':read_logs(200)}



# -----------------------------------------------------------------------------
# V5 conversational Agent: the user sees ONE chat interface.
# The 12-step workflow is hidden backstage. Only an optional audit trace is shown.
# The trace contains tool/status metadata, never private chain-of-thought.
# -----------------------------------------------------------------------------
SESSIONS: dict[str, dict[str, Any]] = {}


def _session(session_id: str | None) -> tuple[str, dict[str, Any]]:
    sid=(session_id or '').strip() or ('sess_'+uuid.uuid4().hex[:12])
    if sid not in SESSIONS:
        SESSIONS[sid]={
            'created_at': now_iso(), 'records': None, 'raw_records': None, 'mapping': None,
            'validation': None, 'snapshot_id': None, 'resolved_input': None,
            'baseline': None, 'scenario': None, 'last_file': None,
            'pending_records': None, 'pending_confirmation': False,
            'messages': [], 'pending_request': None,
            'constraints': {'allow_proxy': None}, 'file_state': {},
            'baseline_explanation': '', 'scenario_explanation': '',
        }
    return sid, SESSIONS[sid]


def _action(label: str, action: str, **kwargs):
    return {'label':label,'action':action,**kwargs}


def _trace(step: str, status: str='done', detail: str=''):
    return {'step':step,'status':status,'detail':detail}


def _ai_meta():
    return {'provider':'DeepSeek' if deepseek_configured() else 'local_fallback',
            'model':deepseek_model() if deepseek_configured() else None,
            'configured':deepseek_configured()}


def _mapping_is_standard(mapping: pd.DataFrame, columns: list[str]) -> bool:
    if mapping is None or mapping.empty: return False
    mp=dict(zip(mapping['system_field'],mapping['source_column']))
    req=bool(mp.get('material')) and bool(mp.get('purchase_weight')) and bool(mp.get('node_id') or mp.get('node_name'))
    # Exact standard template: user has explicitly filled the canonical columns.
    exact={'enterprise','material','node_id','node_name','purchase_weight','year'}.issubset(set(map(str,columns)))
    return req and exact


def _validation_message(validation: dict) -> str:
    st=validation.get('status','unknown')
    issues=validation.get('issues') or []
    gaps=validation.get('data_gaps') or []
    warnings=validation.get('warnings') or []
    parts=[]
    if issues: parts.append('需要确认：'+'；'.join(map(str,issues)))
    if gaps: parts.append('数据缺口：'+'；'.join(map(str,gaps)))
    if warnings: parts.append('提示：'+'；'.join(map(str,warnings)))
    return '\n'.join(parts) or f'数据检查状态：{st}'


def _material_from_records(records: list[dict] | None) -> str:
    if not records: return ''
    vals=[str(x.get('material') or '').strip() for x in records]
    vals=[x for x in vals if x]
    return vals[0] if vals else ''


def _normalize_deepseek_records(obj: dict, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """Convert DeepSeek-extracted candidate records to canonical records without inventing values."""
    records=obj.get('records') or []
    enterprise=str(obj.get('enterprise') or '')
    rows=[]
    for r in records:
        if not isinstance(r,dict): continue
        mat=str(r.get('material') or '').strip()
        nname=str(r.get('node_name') or '').strip()
        nid=str(r.get('node_id') or '').strip()
        w=r.get('purchase_weight')
        try: w=None if w in (None,'') else float(w)
        except Exception: w=None
        # deterministic local resolver may standardize a node name; ambiguous matches remain reviewable.
        if not nid and nname:
            loc=resolve_location(nname, mat or None)
            if loc.get('data') and loc.get('status') in {'success','partial'}:
                nid=loc['data'].get('node_id') or ''
                nname=loc['data'].get('node_name') or nname
        rows.append({
            'enterprise':enterprise,
            'material':mat,
            'node_id':nid,
            'node_name':nname,
            'purchase_weight':w,
            'year':r.get('year') or '',
            'source_type':'V-candidate-ai',
            'confidence':str(r.get('confidence') or 'Low'),
            'evidence':str(r.get('evidence') or '')[:400],
        })
    if rows:
        df=pd.DataFrame(rows)
        # data_tool expects these canonical columns; evidence can remain as extra field in pending data.
        keep=['enterprise','material','node_id','node_name','purchase_weight','year','source_type','confidence']
        return df[keep]
    return fallback_df


def _resolve_registered_data(records: list[dict]) -> tuple[list[dict], str, list[str]]:
    out=[]; warnings=[]; overall='success'
    for row in records:
        r=dict(row)
        node_id=str(r.get('node_id') or '').strip()
        if not node_id and r.get('node_name'):
            loc=resolve_location(str(r.get('node_name')), str(r.get('material') or ''))
            if loc.get('data'):
                node_id=loc['data']['node_id']; r['node_id']=node_id; r['node_name']=loc['data']['node_name']
            if loc.get('status')!='success' and overall!='insufficient': overall='partial'
            warnings.extend(loc.get('warnings',[]))
        wr=lookup_water_risk(node_id)
        mp=lookup_material_parameters(str(r.get('material') or ''))
        if wr.get('status')=='insufficient' or mp.get('status')=='insufficient': overall='insufficient'
        elif wr.get('status')!='success' or mp.get('status')!='success': overall='partial' if overall!='insufficient' else overall
        r['hazard']=wr.get('data')
        r['material_parameters']=mp.get('data')
        warnings.extend(wr.get('warnings',[])); warnings.extend(mp.get('warnings',[]))
        out.append(r)
    return out, overall, list(dict.fromkeys(warnings))


def _update_user_constraints(sess: dict[str,Any], message: str) -> None:
    msg=(message or "").strip()
    if not msg: return
    if re.search(r"(不得|不要|不允许|不能|禁止).{0,12}(代理|假设)",msg):
        sess.setdefault("constraints",{})["allow_proxy"]=False
    elif re.search(r"(允许|可以|接受|同意).{0,12}(代理|假设)",msg):
        sess.setdefault("constraints",{})["allow_proxy"]=True

def _runtime_context(sess: dict[str,Any], stop_reason: str="") -> dict[str,Any]:
    records=sess.get("records") or sess.get("pending_records") or []
    return {
        "file_loaded":bool(sess.get("last_file")), "file_name":sess.get("last_file"),
        "recognized_record_count":len(records), "pending_confirmation":bool(sess.get("pending_confirmation")),
        "constraints":sess.get("constraints") or {}, "stop_reason":stop_reason,
        "validation":sess.get("validation") or {},
    }

def _nonverified_inputs(enriched: list[dict]) -> list[str]:
    found=[]
    for r in enriched or []:
        hz=r.get("hazard") or {}; mp=r.get("material_parameters") or {}
        htype=str(hz.get("data_type") or "")
        if htype and htype!="V": found.append(f"{r.get('node_name') or r.get('node_id')} 地区风险数据={htype}")
        for f,st in (mp.get("data_type") or {}).items():
            st=str(st or "")
            if st and st!="V": found.append(f"{r.get('material')} {f}={st}")
    return list(dict.fromkeys(found))

def _extract_message_records(message: str) -> pd.DataFrame:
    msg=(message or "").strip()
    if not msg: return pd.DataFrame()
    ds=deepseek_extract_supply_chain(msg) if deepseek_configured() else None
    if ds:
        df=_normalize_deepseek_records(ds,pd.DataFrame())
        if not df.empty and df["purchase_weight"].notna().all() and (df["node_id"].astype(str).str.strip()!="").all():
            return df
    mats=[m for m in ["甘蔗","甜菜","大豆"] if m in msg]; mat=mats[0] if len(mats)==1 else ""
    rows=[]
    for mt in re.finditer(r"\b(SC\d{2}|SB\d{2}|SY\d{2})\b.{0,20}?(\d+(?:\.\d+)?)\s*%",msg,re.I):
        nid=mt.group(1).upper(); w=float(mt.group(2))/100.0; loc=resolve_location(nid,mat or None)
        if loc.get("data"):
            rows.append({"enterprise":"","material":mat or loc["data"].get("material",""),"node_id":nid,
                         "node_name":loc["data"].get("node_name",""),"purchase_weight":w,"year":"",
                         "source_type":"V-candidate-text","confidence":"High"})
    return pd.DataFrame(rows)

def _explanation_response(text: str, meta: dict, fallback: str, base_actions: list[dict]) -> tuple[str,str,list[dict]]:
    if meta.get("complete",True): return text,"success",base_actions
    note="\n\n智能解释这次没有完整生成；上面的计算结果已经保留。你可以重试解释，不需要重新计算。"
    return (fallback or text or "计算结果已经保留。")+note,"partial",base_actions+[_action("重试解释","retry_explanation")]

def _auto_analyze(sid: str, sess: dict[str, Any], *, requested_scenario: str|None=None,
                  scenario_params: dict[str,Any]|None=None) -> dict:
    """Hidden backend chain: validate -> trusted data -> current-risk calculation -> optional stress comparison -> AI explanation."""
    trace=[]
    if not sess.get('records'):
        return {'status':'needs_input','message':'可以继续做更贴近你企业实际采购情况的分析。为了判断哪些供应来源最值得关注，我还需要知道原材料来自哪些地区或供应商，以及各自大概占多少。你可以直接上传 ESG 报告、采购表或其他相关文件；没有现成表格也可以使用 Excel 模板。',
                'actions':[_action('上传文件','focus_upload'),_action('下载 Excel 模板','download_template')], 'trace':trace, 'ai':_ai_meta()}

    df=pd.DataFrame(sess['records'])
    validation=validate_normalized(df)
    governance=evaluate_user_records(sess.get('records'))
    validation['field_contract']=governance
    validation['governance_context']=deepseek_context('scenario' if requested_scenario else 'baseline', requested_scenario)
    sess['validation']=jsonable(validation)
    trace.append(_trace('数据完整性检查', validation.get('status','unknown'), _validation_message(validation)))
    trace.append(_trace('检查所需信息', governance.get('status','unknown'), '正在确认这次分析需要的信息是否齐全；缺少关键内容时会先向你询问。'))
    if governance.get('missing_fields'):
        return {'status':'insufficient','message':'我已经检查过你提供的信息，但还缺少这些关键内容：'+ '；'.join(governance.get('missing_details') or governance.get('missing_fields') or []) + '\n\n' + user_requirements_text(),
                'actions':[_action('重新上传/补充数据','focus_upload'),_action('下载 Excel 模板','download_template')], 'trace':trace, 'ai':_ai_meta()}

    if validation.get('issues'):
        return {'status':'needs_confirmation','message':'我已经读完数据，不过有几项关键信息还需要你确认后才能继续：\n' + _validation_message(validation),
                'actions':[_action('重新上传文件','focus_upload'),_action('下载 Excel 模板','download_template')], 'trace':trace, 'ai':_ai_meta()}
    if validation.get('data_gaps'):
        return {'status':'needs_confirmation','message':'我已经识别到你的采购来源，但有些供应地区还无法可靠匹配。请核对地区或供应节点名称后再上传；没有确认的地点我不会当成低风险处理。\n' + _validation_message(validation),
                'actions':[_action('重新上传文件','focus_upload'),_action('下载 Excel 模板','download_template')], 'trace':trace, 'ai':_ai_meta()}

    trace.append(_trace('补充参考信息','running','正在根据供应地区和原材料匹配项目中已核验的参考信息。'))
    enriched,qstatus,qwarn=_resolve_registered_data(sess['records'])
    sess['resolved_input']=jsonable(enriched)
    trace[-1]['status']=qstatus
    trace[-1]['detail']='；'.join(qwarn) if qwarn else '查询完成；来源、数据身份与置信度已保留。'
    if qstatus=='insufficient':
        return {'status':'insufficient','message':'目前还缺少会直接影响结果的关键信息，所以我先不生成正式风险结论。你可以补充对应的供应地区或原材料信息，我会继续完成分析。',
                'actions':[_action('查看需要哪些数据','data_requirements'),_action('重新上传文件','focus_upload')], 'trace':trace, 'ai':_ai_meta()}

    nonverified=_nonverified_inputs(enriched)
    if (sess.get('constraints') or {}).get('allow_proxy') is False and nonverified:
        count=len(sess.get('records') or [])
        fname=sess.get('last_file') or '你提供的数据'
        return {'status':'insufficient',
                'message':f'我已经读取 {fname}，并识别出 {count} 条采购记录。但你明确要求不使用代理值或假设，而当前项目数据中仍有这些已标记内容：'+'；'.join(nonverified[:8])+'。因此我按你的要求停止计算。你可以补充已核验数据，或明确允许使用这些已标记数据后再继续。',
                'actions':[_action('允许使用已标记数据并继续','allow_proxy'),_action('重新上传已核验数据','focus_upload')],
                'trace':trace,'ai':_ai_meta()}

    run_id='run_'+uuid.uuid4().hex[:12]
    sess['snapshot_id']='snap_'+uuid.uuid4().hex[:10]
    trace.append(_trace('计算当前风险','running','正在根据你确认的采购信息和已核验参考数据计算当前风险。'))
    bres=calculate_baseline(df, run_id=run_id)
    sess['baseline']=jsonable(bres)
    trace[-1]['status']=bres.get('status','unknown')
    trace[-1]['detail']='当前风险计算完成；'+str((bres.get('data_identity') or {}).get('label') or '数据身份已记录')+'。'
    log_event({'event':'agent_auto_baseline','request_id':run_id,'session_id':sid,'status':bres.get('status'),'input_hash':bres.get('input_hash'),'model_version':bres.get('model_version')})
    if bres.get('status') in {'insufficient','conflict','error'} or bres.get('data') is None:
        return {'status':bres.get('status'),'message':'目前还无法形成可用的风险结果。'+('；'.join(map(str,bres.get('data_gaps') or [])) or '请检查你提供的信息是否完整。'),
                'actions':[_action('重新上传/补充数据','focus_upload')], 'trace':trace, 'ai':_ai_meta()}

    if requested_scenario:
        mat=_material_from_records(sess.get('records'))
        sp=scenario_params or {}
        trace.append(_trace('比较不同情况','running','正在比较你选择的压力情况下风险和供应变化。'))
        sres=run_scenario(
            restore_result(sess['baseline']), requested_scenario, mat,
            int(sp.get('year') or 2050), str(sp.get('path') or 'BAU'),
            float(sp.get('failure_fraction') if sp.get('failure_fraction') is not None else 1.0),
            float(sp.get('inventory') if sp.get('inventory') is not None else 0.10),
        )
        sess['scenario']=jsonable(sres)
        trace[-1]['status']=sres.get('status','unknown')
        trace[-1]['detail']='；'.join(sres.get('warnings') or []) or '不同情况的比较已完成。'
        log_event({'event':'agent_auto_scenario','request_id':run_id,'session_id':sid,'scenario_type':requested_scenario,'status':sres.get('status'),'baseline_input_hash':bres.get('input_hash')})
        if sres.get('data') is not None:
            fallback=scenario_brief(restore_result(sess['scenario']))
            text,meta=analyze_tool_result(
                user_question='请用普通用户能听懂的话解释这种情况下相对当前情况发生了什么变化、对供应意味着什么，以及建议怎么做。',
                baseline=sess['baseline'], scenario=sess['scenario'], validation=sess.get('validation'),
                runtime_context=_runtime_context(sess), fallback=fallback)
            base_actions=[_action('查看当前分析结果','show_current'),_action('换一种情况','scenario_menu'),_action('导出完整报告','export_report')]
            text,status,actions=_explanation_response(text,meta,fallback,base_actions)
            sess['scenario_explanation']=text
            return {'status':status,'message':text,'actions':actions,
                    'trace':trace,'ai':meta,'result_summary':{'baseline':sess['baseline'],'scenario':sess['scenario']}}


    fallback=baseline_brief(restore_result(sess['baseline']))
    text,meta=analyze_tool_result(
        user_question='请用普通用户能听懂的话解释当前风险结果、主要风险来源、重点供应地和管理建议。',
        baseline=sess['baseline'], validation=sess.get('validation'), runtime_context=_runtime_context(sess), fallback=fallback)
    base_actions=[_action('关键用水期压力升高','run_scenario',scenario_type='PeakSeason'),
                  _action('未来水环境发生变化','run_scenario',scenario_type='AqueductFuture'),
                  _action('严重干旱再次发生','run_scenario',scenario_type='ExtremeDrought'),
                  _action('主要供应节点中断','run_scenario',scenario_type='NodeFailure'),
                  _action('导出当前报告','export_report')]
    text,status,actions=_explanation_response(text,meta,fallback,base_actions)
    sess['baseline_explanation']=text
    if status=='success':
        text += '\n\n如果你想继续，我还可以帮你看看在不同压力情况下，风险和供应可能会怎么变化。'
    return {'status':status,'message':text,'actions':actions,'trace':trace,'ai':meta,'result_summary':{'baseline':sess['baseline']}}


@app.get('/api/agent/status')
def agent_status():
    return {'status':'ok','agent_version':'9.2-audit-fixed',
            'deepseek_configured':deepseek_configured(),
            'deepseek_env_source':deepseek_key_source(),
            'deepseek_model':deepseek_model(),
            'deepseek_base_url':deepseek_base_url(),
            'field_contract':contract_summary(),
            'message':'DeepSeek 已配置' if deepseek_configured() else 'DeepSeek 尚未检测到密钥；当前仍可使用本地说明。'}


@app.get('/api/template')
def template_download():
    p=ROOT/'static'/'Water_Risk_Input_Template.xlsx'
    if not p.exists(): raise HTTPException(404,'模板不存在')
    return FileResponse(p,filename='Water_Risk_Input_Template.xlsx',media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.post('/api/agent/message')
async def agent_message(
    message: str = Form(''), session_id: str = Form(''), action: str = Form(''),
    scenario_type: str = Form(''), material: str = Form(''),
    year: str = Form(''), path: str = Form(''), failure_fraction: str = Form(''), inventory: str = Form(''),
    file: UploadFile | None = File(None)
):
    sid,sess=_session(session_id); trace=[]; msg=(message or '').strip()
    _update_user_constraints(sess,msg)
    sess['messages'].append({'role':'user','content':msg,'timestamp':now_iso()}) if msg else None
    # Understand the user's request before processing the uploaded file, but keep this hidden from the UI.
    early_intent=classify_intent(msg) if msg else None
    early_rule=route_intent(msg) if msg else {'intent':'general','scenario_type':None}
    pending_intent=(early_intent or {}).get('intent') or ({'baseline':'analyze','scenario':'scenario','data_audit':'data_help','explain':'explain'}.get(early_rule.get('intent'),'general'))
    pending_scenario=(early_intent or {}).get('scenario_type') or early_rule.get('scenario_type')
    pending_params={
        'year':(early_intent or {}).get('year') or 2050, 'path':(early_intent or {}).get('path') or 'BAU',
        'failure_fraction':(early_intent or {}).get('failure_fraction'), 'inventory':(early_intent or {}).get('inventory'),
    }
    if msg:
        sess['pending_request']={'intent':pending_intent,'scenario_type':pending_scenario,'params':pending_params,'message':msg}

    # File input is interpreted inside the conversation; no separate workflow page exists.
    if file is not None and file.filename:
        suffix=Path(file.filename).suffix.lower()
        if suffix not in ['.csv','.xlsx','.xls','.docx','.pdf']:
            return {'session_id':sid,'status':'error','message':'目前支持 PDF、DOCX、XLSX、XLS、CSV。','actions':[],'trace':[],'ai':_ai_meta()}
        payload=await file.read(); tmp=ROOT/'outputs'/f'upload_{uuid.uuid4().hex[:10]}{suffix}'; tmp.write_bytes(payload)
        try:
            res=read_user_file(str(tmp)); df=res.get('data') if isinstance(res.get('data'),pd.DataFrame) else pd.DataFrame()
            sess['last_file']=file.filename
            sess['file_state']={'loaded':True,'file_name':file.filename,'kind':res.get('kind'),'initial_candidate_count':len(df)}
            trace.append(_trace('读取文件','done',f'{file.filename} · {res.get("kind")} · {len(df)} 条初始候选记录'))
            if res.get('kind')=='table':
                mp=suggest_mapping(list(df.columns)); sess['mapping']=jsonable(mp); sess['raw_records']=jsonable(df)
                trace.append(_trace('字段识别','done','识别材料、供应节点/地区、采购权重和年份等计算字段。'))
                mpd=dict(zip(mp['system_field'],mp['source_column']))
                if _mapping_is_standard(mp,list(df.columns)):
                    norm=apply_mapping(df,mp,'0-1'); sess['records']=jsonable(norm); sess['pending_confirmation']=False
                    trace.append(_trace('标准模板确认','done','检测到标准字段，按用户上传的标准模板直接进入自动分析。'))
                elif mpd.get('material') and mpd.get('purchase_weight') and (mpd.get('node_id') or mpd.get('node_name')):
                    norm=apply_mapping(df,mp,'0-1'); sess['pending_records']=jsonable(norm); sess['pending_confirmation']=True
                    mapping_text='；'.join(f"{r['source_column']} → {r['system_field']}" for _,r in mp.iterrows() if r['source_column'])
                    return {'session_id':sid,'status':'needs_confirmation',
                            'message':'我已经读完表格，并识别出关键字段。为了避免把“采购占比”等列理解错，请确认这次字段对应关系：\n'+mapping_text,
                            'actions':[_action('确认并继续分析','confirm_mapping'),_action('改用标准模板','download_template')],
                            'trace':trace,'ai':_ai_meta()}
                else:
                    return {'session_id':sid,'status':'needs_input',
                            'message':'文件已经读完，不过还没有同时找到“原材料、供应来源/地区、各来源采购量或占比”。你可以补充这些信息，或者下载 Excel 模板填写后重新上传。',
                            'actions':[_action('下载 Excel 模板','download_template'),_action('重新上传','focus_upload')],
                            'trace':trace,'ai':_ai_meta()}
            else:
                # For ESG/PDF/Word, DeepSeek performs candidate extraction when configured.
                text=(res.get('text') or '')
                fallback_df=df
                ds=deepseek_extract_supply_chain(text)
                norm=_normalize_deepseek_records(ds, fallback_df) if ds else fallback_df
                sess['raw_records']=jsonable(norm)
                sess['pending_records']=jsonable(norm)
                sess['pending_confirmation']=True
                trace.append(_trace('报告信息抽取','done',f'生成 {len(norm)} 条供应链候选记录'+('（DeepSeek）' if ds else '（本地规则回退）')+'。'))
                if norm.empty:
                    return {'session_id':sid,'status':'needs_input',
                            'message':'我已经读完这份报告，但里面还没有足够的信息来判断你企业的实际采购风险。你可以再上传采购表，或者下载 Excel 模板补充原材料、供应来源和采购占比。',
                            'actions':[_action('上传采购表','focus_upload'),_action('下载 Excel 模板','download_template')],
                            'trace':trace,'ai':_ai_meta()}
                # Always confirm document extraction because it is candidate data.
                preview=[]
                for r in norm.to_dict(orient='records')[:8]:
                    preview.append(f"{r.get('material') or '材料待确认'} / {r.get('node_name') or r.get('node_id') or '地点待确认'} / 权重 {r.get('purchase_weight') if r.get('purchase_weight') not in (None,'') else '未找到'}")
                return {'session_id':sid,'status':'needs_confirmation',
                        'message':'我从报告中提取到了以下供应链候选信息。报告抽取属于候选数据，需要你确认后我才能自动进入正式计算：\n'+'\n'.join('• '+x for x in preview),
                        'actions':[_action('确认候选信息并继续','confirm_mapping'),_action('上传更完整的采购表','focus_upload'),_action('下载 Excel 模板','download_template')],
                        'trace':trace,'ai':_ai_meta()}
        finally:
            try: tmp.unlink(missing_ok=True)
            except Exception: pass

    if action=='confirm_mapping':
        if sess.get('pending_records'):
            sess['records']=sess.pop('pending_records'); sess['pending_confirmation']=False
            trace.append(_trace('人工确认','done','关键字段/候选信息已确认，仅用于本次分析。'))
            pr=sess.get('pending_request') or {}
            req_scenario=pr.get('scenario_type') if pr.get('intent')=='scenario' else None
            out=_auto_analyze(sid,sess,requested_scenario=req_scenario,scenario_params=pr.get('params') or {})
            out['trace']=trace+out.get('trace',[]); out['session_id']=sid
            return out
        return {'session_id':sid,'status':'needs_input','message':'当前没有等待确认的数据，请先上传文件。','actions':[_action('上传文件','focus_upload')],'trace':trace,'ai':_ai_meta()}

    if action=='use_demo':
        mat=material or '甘蔗'
        if mat not in ['甘蔗','甜菜','大豆']: mat='甘蔗'
        ddf=load_demo_procurement(mat); sess['records']=jsonable(ddf)
        trace.append(_trace('载入示范案例','done',f'{mat}；采购权重为研究假设，仅用于演示。'))
        out=_auto_analyze(sid,sess)
        out['message']='**演示说明：本次采购结构为研究假设，不代表真实企业采购。**\n\n'+out.get('message','')
        out['trace']=trace+out.get('trace',[]); out['session_id']=sid
        return out

    if action=='allow_proxy':
        sess.setdefault('constraints',{})['allow_proxy']=True
        out=_auto_analyze(sid,sess); out['session_id']=sid; return out

    if action=='retry_explanation':
        if sess.get('scenario'):
            fallback=scenario_brief(restore_result(sess['scenario']))
            text,meta=analyze_tool_result(user_question='请重新完整解释这次不同情况比较，并给出完整建议。',
                baseline=sess.get('baseline'),scenario=sess.get('scenario'),validation=sess.get('validation'),runtime_context=_runtime_context(sess),fallback=fallback)
            text,status,actions=_explanation_response(text,meta,fallback,[_action('换一种情况','scenario_menu'),_action('导出完整报告','export_report')])
            sess['scenario_explanation']=text
            return {'session_id':sid,'status':status,'message':text,'actions':actions,'trace':trace,'ai':meta,'result_summary':{'baseline':sess.get('baseline'),'scenario':sess.get('scenario')}}
        if sess.get('baseline'):
            fallback=baseline_brief(restore_result(sess['baseline']))
            text,meta=analyze_tool_result(user_question='请重新完整解释当前风险结果，并给出完整建议。',
                baseline=sess.get('baseline'),validation=sess.get('validation'),runtime_context=_runtime_context(sess),fallback=fallback)
            text,status,actions=_explanation_response(text,meta,fallback,[_action('看看不同情况下会怎样','scenario_menu'),_action('导出报告','export_report')])
            sess['baseline_explanation']=text
            return {'session_id':sid,'status':status,'message':text,'actions':actions,'trace':trace,'ai':meta,'result_summary':{'baseline':sess.get('baseline')}}

    if action=='run_scenario':
        sp={
            'year': int(year) if str(year).isdigit() else 2050,
            'path': path or 'BAU',
            'failure_fraction': float(failure_fraction) if str(failure_fraction).strip() else 1.0,
            'inventory': float(inventory) if str(inventory).strip() else 0.10,
        }
        if not sess.get('baseline'):
            out=_auto_analyze(sid,sess,requested_scenario=scenario_type or 'NodeFailure',scenario_params=sp)
            out['session_id']=sid; return out
        b=restore_result(sess['baseline']); mat=_material_from_records(sess.get('records'))
        st=scenario_type or 'NodeFailure'
        trace.append(_trace('比较不同情况','running','正在比较你选择的压力情况下风险和供应变化。'))
        sres=run_scenario(b,st,mat,sp['year'],sp['path'],sp['failure_fraction'],sp['inventory']); sess['scenario']=jsonable(sres)
        trace[-1]['status']=sres.get('status','unknown'); trace[-1]['detail']='；'.join(sres.get('warnings') or []) or '计算完成。'
        if sres.get('data') is None:
            return {'session_id':sid,'status':sres.get('status'),'message':'这种情况暂时无法完成分析：'+'；'.join(map(str,sres.get('data_gaps') or [])), 'actions':[_action('换一种情况','scenario_menu')], 'trace':trace,'ai':_ai_meta()}
        fallback=scenario_brief(restore_result(sess['scenario']))
        text,meta=analyze_tool_result(user_question=msg or '请完整解释这种情况下发生了什么变化。', baseline=sess.get('baseline'),
                                      scenario=sess.get('scenario'), validation=sess.get('validation'), runtime_context=_runtime_context(sess), fallback=fallback)
        text,status,actions=_explanation_response(text,meta,fallback,[_action('换一种情况','scenario_menu'),_action('导出完整报告','export_report')])
        sess['scenario_explanation']=text
        return {'session_id':sid,'status':status,'message':text,'actions':actions,'trace':trace,'ai':meta,
                'result_summary':{'baseline':sess.get('baseline'),'scenario':sess.get('scenario')}}

    if action=='show_current' and sess.get('baseline'):
        fallback=baseline_brief(restore_result(sess['baseline']))
        text,meta=analyze_tool_result(user_question='请用普通用户能听懂的话解释当前风险结果、重点供应地、主要原因和建议。', baseline=sess.get('baseline'),
                                      validation=sess.get('validation'), runtime_context=_runtime_context(sess), fallback=fallback)
        text,status,actions=_explanation_response(text,meta,fallback,[_action('看看不同情况下会怎样','scenario_menu'),_action('导出报告','export_report')])
        sess['baseline_explanation']=text
        return {'session_id':sid,'status':status,'message':text,'actions':actions, 'trace':trace,'ai':meta,
                'result_summary':{'baseline':sess.get('baseline')}}

    if action=='scenario_menu':
        return {'session_id':sid,'status':'success','message':'可以。你还想看看哪一种情况下风险会怎样变化？',
                'actions':[_action('关键用水期压力升高','run_scenario',scenario_type='PeakSeason'),_action('未来水环境发生变化','run_scenario',scenario_type='AqueductFuture'),_action('严重干旱再次发生','run_scenario',scenario_type='ExtremeDrought'),_action('主要供应节点中断','run_scenario',scenario_type='NodeFailure')],
                'trace':trace,'ai':_ai_meta()}

    if action=='data_requirements':
        return {'session_id':sid,'status':'success','message':user_requirements_text(),
                'actions':[_action('下载 Excel 模板','download_template'),_action('上传文件','focus_upload')], 'trace':trace,'ai':_ai_meta(),
                'field_contract':contract_summary()}

    if action=='export_report':
        if not sess.get('baseline'):
            return {'session_id':sid,'status':'needs_input','message':'当前还没有可导出的完整分析结果。请先完成一次风险分析。','actions':[],'trace':trace,'ai':_ai_meta()}
        return {'session_id':sid,'status':'export_ready','message':'可以，报告已经准备好。',
                'actions':[_action('下载 Word 分析报告','download_report')],'trace':trace,'ai':_ai_meta()}

    # DeepSeek handles natural-language intent and scenario parameter extraction; deterministic router remains fallback.
    deep_intent=classify_intent(msg) if msg else None
    rule=route_intent(msg)
    intent=(deep_intent or {}).get('intent') or ({'baseline':'analyze','scenario':'scenario','data_audit':'data_help','explain':'explain'}.get(rule.get('intent'),'general'))
    requested=(deep_intent or {}).get('scenario_type') or rule.get('scenario_type')
    sp={
        'year':(deep_intent or {}).get('year') or 2050,
        'path':(deep_intent or {}).get('path') or 'BAU',
        'failure_fraction':(deep_intent or {}).get('failure_fraction'),
        'inventory':(deep_intent or {}).get('inventory'),
    }
    if msg:
        trace.append(_trace('理解需求','done',f'intent={intent}'+(f' · scenario={requested}' if requested else '')))

    if intent=='export':
        if sess.get('baseline'):
            return {'session_id':sid,'status':'export_ready','message':'本次分析可以导出。', 'actions':[_action('下载 Word 分析报告','download_report')], 'trace':trace,'ai':_ai_meta()}
        return {'session_id':sid,'status':'needs_input','message':'还没有可导出的完整分析结果。你可以先上传相关资料，我会继续完成分析。','actions':[_action('上传文件','focus_upload')],'trace':trace,'ai':_ai_meta()}

    if sess.get('records') and intent in {'analyze','scenario'}:
        out=_auto_analyze(sid,sess,requested_scenario=requested if intent=='scenario' else None,scenario_params=sp)
        out['trace']=trace+out.get('trace',[]); out['session_id']=sid
        return out

    if sess.get('scenario') and msg:
        fallback=scenario_brief(restore_result(sess['scenario']))
        ans,meta=analyze_tool_result(user_question=msg, baseline=sess.get('baseline'), scenario=sess.get('scenario'),
                                     validation=sess.get('validation'), runtime_context=_runtime_context(sess), fallback=fallback)
        ans,status,actions=_explanation_response(ans,meta,fallback,[_action('换一种情况','scenario_menu'),_action('导出完整报告','export_report')])
        sess['scenario_explanation']=ans
        return {'session_id':sid,'status':status,'message':ans,'actions':actions,'trace':trace,'ai':meta,
                'result_summary':{'baseline':sess.get('baseline'),'scenario':sess.get('scenario')}}
    if sess.get('baseline') and msg:
        fallback=baseline_brief(restore_result(sess['baseline']))
        ans,meta=analyze_tool_result(user_question=msg, baseline=sess.get('baseline'), validation=sess.get('validation'), runtime_context=_runtime_context(sess), fallback=fallback)
        ans,status,actions=_explanation_response(ans,meta,fallback,[_action('看看不同情况下会怎样','scenario_menu'),_action('导出报告','export_report')])
        sess['baseline_explanation']=ans
        return {'session_id':sid,'status':status,'message':ans,'actions':actions,'trace':trace,'ai':meta,
                'result_summary':{'baseline':sess.get('baseline')}}

    if msg and not sess.get('records') and intent=='analyze':
        cand=_extract_message_records(msg)
        if not cand.empty:
            sess['pending_records']=jsonable(cand); sess['pending_confirmation']=True
            preview=[f"{r.get('material') or '材料待确认'} / {r.get('node_name') or r.get('node_id')} / {float(r.get('purchase_weight'))*100:.1f}%" for r in cand.to_dict(orient='records')]
            return {'session_id':sid,'status':'needs_confirmation','message':'我从你的文字中识别到以下采购结构。请确认后我再继续：\n'+'\n'.join('• '+x for x in preview),
                    'actions':[_action('确认并继续分析','confirm_mapping'),_action('上传采购表','focus_upload')],'trace':trace,'ai':_ai_meta()}

    # No enterprise dataset yet: DeepSeek should still answer the user's question in plain language first,
    # then naturally explain which information would make the analysis company-specific.
    if msg:
        fallback=(
            '可以，我先从你的问题本身开始分析。上游原材料的水风险通常要同时看供应地区是否长期缺水、是否容易发生干旱或季节性供水波动，'
            '以及你的采购是否集中在少数来源。\n\n'
            '如果你希望进一步得到与你企业实际采购结构相关的结果，我还需要知道原材料主要从哪些地区或供应商采购、各自大概占多少。'
            '你可以直接上传 ESG 报告或采购表；如果没有现成表格，也可以下载 Excel 模板填写。'
        )
        task_context='scenario' if intent=='scenario' else 'current_risk'
        ans,meta=general_guidance(msg,fallback,deepseek_context(task_context, requested))
        return {'session_id':sid,'status':'success','message':ans,
                'actions':[_action('上传相关资料','focus_upload'),_action('下载 Excel 模板','download_template')],
                'trace':trace,'ai':meta}

    return {'session_id':sid,'status':'needs_input',
            'message':'你好，我是 WaterPulse。你可以直接告诉我想了解的原材料、供应地区或风险问题，也可以上传 ESG 报告或采购资料。我会先理解你的需求，再告诉你下一步需要什么。',
            'actions':[_action('上传相关资料','focus_upload'),_action('下载 Excel 模板','download_template')],
            'trace':trace,'ai':_ai_meta()}


@app.get('/api/supported-locations')
def api_supported_locations(material: str=''):
    return supported_locations(material or None)

@app.get('/api/agent/session/{session_id}')
def agent_session_state(session_id: str):
    sess=SESSIONS.get(session_id)
    if not sess: raise HTTPException(404,'会话不存在或服务已重启')
    return {'status':'success','session_id':session_id,'file_state':sess.get('file_state') or {},
            'validation':sess.get('validation'),'baseline':sess.get('baseline'),'scenario':sess.get('scenario'),
            'constraints':sess.get('constraints') or {},'baseline_explanation':sess.get('baseline_explanation') or '',
            'scenario_explanation':sess.get('scenario_explanation') or ''}

@app.get('/api/data-contract')
def data_contract():
    return {'status':'success','summary':contract_summary(),'requirements':user_requirements_text(),'context':deepseek_context('current_risk')}


@app.get('/api/agent/report/{session_id}')
def agent_report(session_id: str):
    sess=SESSIONS.get(session_id)
    if not sess or not sess.get('baseline'):
        raise HTTPException(404,'该会话暂无可导出的报告')
    b=restore_result(sess['baseline']); s=restore_result(sess.get('scenario'))
    path=export_risk_report(b,s,sess.get('validation'),sess.get('snapshot_id') or session_id,sess.get('baseline_explanation') or '',sess.get('scenario_explanation') or '')
    log_event({'event':'agent_export','session_id':session_id,'snapshot_id':sess.get('snapshot_id'),'input_hash':(sess.get('baseline') or {}).get('input_hash')})
    return FileResponse(path,filename=Path(path).name,media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')


# static must be mounted last so API routes take precedence
app.mount('/', StaticFiles(directory=ROOT/'static', html=True), name='static')
