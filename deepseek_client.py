from __future__ import annotations
import json
import os
import re
from typing import Any
import requests

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Railway/Linux environment variable names are case-sensitive. Accept the
# canonical name plus common aliases so a valid key is not missed simply
# because the variable name was entered slightly differently.
_KEY_ALIASES = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_KEY",
    "DEEPSEEK_APIKEY",
    "DEEPSEEK_TOKEN",
    "DEEPSEEK_API_TOKEN",
)

def _clean_env_value(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    if "=" in value and value.split("=", 1)[0].strip().upper() in _KEY_ALIASES:
        value = value.split("=", 1)[1].strip()
    return value

def _normalized_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")

def api_key_info() -> tuple[str, str | None]:
    for name in _KEY_ALIASES:
        key = _clean_env_value(os.getenv(name))
        if key:
            return key, name
    wanted = {_normalized_name(x) for x in _KEY_ALIASES}
    for name, value in os.environ.items():
        if _normalized_name(name) in wanted:
            key = _clean_env_value(value)
            if key:
                return key, name
    return "", None

def api_key_value() -> str:
    return api_key_info()[0]

def api_key_source() -> str | None:
    return api_key_info()[1]

def configured() -> bool:
    return bool(api_key_value())

def model_name() -> str:
    value = _clean_env_value(os.getenv("DEEPSEEK_MODEL"))
    return value or DEFAULT_MODEL

def base_url() -> str:
    value = _clean_env_value(os.getenv("DEEPSEEK_BASE_URL"))
    return (value or DEFAULT_BASE_URL).rstrip("/")


def _post_result(messages: list[dict[str, Any]], *, temperature: float = 0.15, max_tokens: int = 2200, json_mode: bool = False) -> dict[str, Any]:
    key = api_key_value()
    if not key:
        raise RuntimeError("DeepSeek API key is not configured in Railway Variables")
    payload: dict[str, Any] = {
        "model": model_name(), "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False, "thinking": {"type": "disabled"},
    }
    if json_mode:
        payload["response_format"]={"type":"json_object"}
    resp=requests.post(base_url()+"/chat/completions",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json=payload,timeout=90)
    resp.raise_for_status()
    obj=resp.json(); choices=obj.get("choices") or []
    if not choices:
        return {"text":"","finish_reason":"missing_choice","usage":obj.get("usage") or {}}
    ch=choices[0] or {}; msg=ch.get("message") or {}
    return {"text":(msg.get("content") or "").strip(),"finish_reason":ch.get("finish_reason"),"usage":obj.get("usage") or {}}

def _post(messages: list[dict[str, Any]], *, temperature: float = 0.15, max_tokens: int = 2200, json_mode: bool = False) -> str:
    return _post_result(messages,temperature=temperature,max_tokens=max_tokens,json_mode=json_mode)["text"]


def _looks_complete(text: str, kind: str="general") -> bool:
    t=(text or "").strip()
    if len(t)<40: return False
    if kind=="baseline": return len(t)>=120 and ("风险" in t) and ("建议" in t) and (("供应地" in t) or ("供应" in t))
    if kind=="scenario": return len(t)>=100 and (("变化" in t) or ("供应" in t)) and ("建议" in t)
    return True

def chat(system_prompt: str, user_prompt: str, fallback: str) -> tuple[str, dict[str, Any]]:
    if not configured():
        return fallback,{"provider":"local_fallback","model":None,"used":False,"complete":True}
    try:
        res=_post_result([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],max_tokens=2200)
        text=_plain_user_language(res.get("text") or "")
        if res.get("finish_reason")=="length" or not _looks_complete(text):
            res=_post_result([{"role":"system","content":system_prompt},{"role":"user","content":user_prompt+"\n\n请完整回答并自然收尾，不要只写半句。"}],max_tokens=3200)
            text=_plain_user_language(res.get("text") or "")
        if res.get("finish_reason")!="stop" or not _looks_complete(text):
            return fallback,{"provider":"local_fallback","model":model_name(),"used":False,"complete":False,"retryable":True,"finish_reason":res.get("finish_reason"),"error":"AI explanation incomplete"}
        return text,{"provider":"DeepSeek","model":model_name(),"used":True,"complete":True,"finish_reason":res.get("finish_reason")}
    except Exception as exc:
        return fallback,{"provider":"local_fallback","model":model_name(),"used":False,"complete":False,"retryable":True,"error":str(exc)[:300]}


def _json_from_text(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def classify_intent(message: str) -> dict[str, Any] | None:
    """Use DeepSeek only for intent/parameter understanding, never for risk calculation."""
    if not configured() or not (message or "").strip():
        return None
    sys = (
        "你是上游供应链水风险 AI Agent 的任务路由器。只做任务识别和情景参数抽取，不计算任何风险数字。"
        "只输出JSON对象，不要Markdown。"
        "intent只能是 analyze、scenario、explain、data_help、export、general。"
        "scenario_type只能是 PeakSeason、AqueductFuture、ExtremeDrought、NodeFailure 或 null。"
        "可选字段：material(甘蔗/甜菜/大豆或null)、year(整数或null)、path(BAU/OPT/PES或null)、"
        "failure_fraction(0-1或null)、inventory(0-1或null)。"
        "如果用户说分析上传材料或当前风险，intent=analyze；如果问缺什么数据或模板，intent=data_help；"
        "如果明确问情景变化，intent=scenario；如果要求下载报告，intent=export。"
    )
    try:
        raw = _post([
            {"role": "system", "content": sys},
            {"role": "user", "content": message},
        ], temperature=0.0, max_tokens=500, json_mode=True)
        obj = _json_from_text(raw)
        if not obj or obj.get("intent") not in {"analyze", "scenario", "explain", "data_help", "export", "general"}:
            return None
        if obj.get("scenario_type") not in {None, "PeakSeason", "AqueductFuture", "ExtremeDrought", "NodeFailure"}:
            obj["scenario_type"] = None
        return obj
    except Exception:
        return None


def extract_supply_chain(text: str) -> dict[str, Any] | None:
    """Extract candidate enterprise/material/node/procurement fields from report text.
    Output is candidate data only and must be confirmed before deterministic calculation.
    """
    if not configured() or not (text or "").strip():
        return None
    excerpt = text[:45000]
    sys = (
        "你是供应链资料结构化抽取工具。只从用户原文抽取，不猜测、不补齐。"
        "请只输出JSON，不要Markdown。结构：{enterprise:null或字符串, records:[...] }。"
        "每条records字段：material,node_id,node_name,purchase_weight,year,evidence,confidence。"
        "material只在原文明确时填甘蔗/甜菜/大豆或原材料原名；purchase_weight统一为0-1，原文没有就null；"
        "node_id只有原文明确包含项目节点编号时填写，否则null；node_name填写供应地/省州/国家/供应节点原文；"
        "evidence必须是支持该条记录的简短原文片段；confidence只可High/Medium/Low。"
        "不要把国家均值、推测值或常识当作企业采购事实。"
    )
    try:
        raw = _post([
            {"role": "system", "content": sys},
            {"role": "user", "content": "请抽取以下报告中的上游采购/供应链候选信息：\n" + excerpt},
        ], temperature=0.0, max_tokens=2200, json_mode=True)
        obj = _json_from_text(raw)
        if not obj or not isinstance(obj.get("records"), list):
            return None
        return obj
    except Exception:
        return None


def general_guidance(message: str, fallback: str, governance_context: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """First-contact conversation. Answer the user's need first, then guide data collection in plain language."""
    sys = (
        "你是 WaterPulse，一名面向科技型农食企业的上游供应链水风险助手。"
        "你的第一任务是先理解并回应用户真正关心的问题，而不是一上来要求上传文件。"
        "如果用户尚未提供企业数据，可以先用定性、易懂的方式说明应该关注哪些水风险来源、为什么值得关注，以及分析思路；"
        "随后再自然说明：如果要得到与该企业采购结构相关的个性化结果，需要补充原材料、主要供应地区/供应商、各来源采购量或占比，最好再有年份。"
        "用户可以上传 ESG 报告、采购表或其他相关文件；没有现成表格时，可以使用系统提供的 Excel 模板。"
        "如果用户问‘不同情况下会怎样’，用日常语言说明可以进一步比较关键用水期、未来水环境变化、严重干旱重现或主要供应节点中断。"
        "面向普通用户时始终使用自然中文，不要出现 Baseline、Scenario、PRWI、WS、DR、SV、BWD、DYS、CTS、OA、JSON、Schema、工具调用、字段合同、工作流、确定性引擎等内部术语。后台计算请转述为“当前风险分析”“不同情况比较”“综合风险指数”“参考数据”等普通表达。"
        "不要凭空生成企业采购比例、地点风险分值、综合风险数值或精确损失数字。"
        "回答要自然、简洁，像真正的企业分析助手；优先用用户能听懂的中文。"
    )
    prompt = message
    if governance_context:
        prompt += '\n\n以下是后台数据要求摘要，只用于判断需要向用户补充哪些信息；不要把内部字段名或技术术语直接展示给用户：\n' + json.dumps(governance_context, ensure_ascii=False, default=str)
    return chat(sys, prompt, fallback)


def _compact_tool_payload(baseline: dict|None, scenario: dict|None, validation: dict|None, runtime_context: dict|None) -> dict[str, Any]:
    out={"runtime_context":runtime_context or {},"validation":validation or {}}
    if baseline:
        bd=baseline.get("data") or {}; summary=bd.get("summary") or []
        if hasattr(summary,"to_dict"): summary=summary.to_dict(orient="records")
        nodes=bd.get("nodes") or []
        if hasattr(nodes,"to_dict"): nodes=nodes.to_dict(orient="records")
        if isinstance(nodes,list): nodes=sorted(nodes,key=lambda x:float(x.get("C") or -1),reverse=True)[:6]
        out["current_risk"]={
            "status":baseline.get("status"),"summary":summary,"top_nodes":nodes,
            "warnings":baseline.get("warnings") or [],"proxy_fields":baseline.get("proxy_fields") or [],
            "assumptions":baseline.get("assumptions") or [],"source_refs":baseline.get("source_refs") or [],
            "data_identity":baseline.get("data_identity") or {},
        }
    if scenario:
        sd=scenario.get("data") or {}; clean={}
        for k,v in sd.items():
            if k in {"node_table","replacement_table"} and hasattr(v,"to_dict"): v=v.to_dict(orient="records")
            clean[k]=(v[:6] if isinstance(v,list) else v)
        out["comparison"]={"status":scenario.get("status"),"data":clean,"warnings":scenario.get("warnings") or []}
    return out

def analyze_tool_result(*, user_question: str, baseline: dict | None = None, scenario: dict | None = None,
                        validation: dict | None = None, runtime_context: dict|None=None, fallback: str = "") -> tuple[str, dict[str, Any]]:
    if not configured(): return fallback,{"provider":"local_fallback","model":None,"used":False,"complete":True}
    payload=_compact_tool_payload(baseline,scenario,validation,runtime_context); payload["user_question"]=user_question
    kind="scenario" if scenario else "baseline"
    sys=(
        "你是 WaterPulse 上游供应链水风险决策助手。runtime_context 是本次真实运行状态，优先级最高；绝不能说没有收到文件、没有识别记录，除非 runtime_context 明确这样写。"
        "你只能解释给定的计算结果，不能重新计算或修改任何数值。没有的数字就说未提供，绝对不要猜。"
        "每次成功解释必须完整包含：①风险值及其含义；②对企业影响最大的供应地；③主要风险来源；④数据缺口、代理或假设；⑤与结果对应的管理建议。"
        "如果是不同情况比较，还必须说清发生了什么变化以及供应含义。"
        "面向普通用户，不要出现 Baseline、Scenario、PRWI、WS、DR、SV、BWD、DYS、CTS、OA、JSON、Schema 等内部术语。"
        "控制在 250-650 字，句子必须写完整并自然收尾。"
    )
    raw_payload=json.dumps(payload,ensure_ascii=False,default=str)
    try:
        messages=[{"role":"system","content":sys},{"role":"user","content":raw_payload}]
        res=_post_result(messages,temperature=0.1,max_tokens=1800); text=_plain_user_language(res.get("text") or "")
        if res.get("finish_reason")=="length" or not _looks_complete(text,kind):
            messages[-1]["content"] += "\n\n上一版可能不完整。请严格按五项要求完整输出，最后必须给出完整管理建议，不要半句结束。"
            res=_post_result(messages,temperature=0.05,max_tokens=2800); text=_plain_user_language(res.get("text") or "")
        if res.get("finish_reason")!="stop" or not _looks_complete(text,kind):
            return fallback,{"provider":"local_fallback","model":model_name(),"used":False,"complete":False,"retryable":True,"finish_reason":res.get("finish_reason"),"error":"AI explanation incomplete or truncated"}
        return text,{"provider":"DeepSeek","model":model_name(),"used":True,"complete":True,"finish_reason":res.get("finish_reason")}
    except Exception as exc:
        return fallback,{"provider":"local_fallback","model":model_name(),"used":False,"complete":False,"retryable":True,"error":str(exc)[:300]}


def connection_test() -> dict[str, Any]:
    """Make a minimal live API call without ever returning the secret itself."""
    source = api_key_source()
    if not configured():
        return {
            "ok": False,
            "configured": False,
            "env_source": None,
            "model": model_name(),
            "base_url": base_url(),
            "message": "No DeepSeek key was found in the service environment.",
        }
    try:
        text = _post([
            {"role": "system", "content": "You are a connection test. Reply with exactly OK."},
            {"role": "user", "content": "ping"},
        ], temperature=0.0, max_tokens=8)
        return {
            "ok": True,
            "configured": True,
            "env_source": source,
            "model": model_name(),
            "base_url": base_url(),
            "reply": text[:50],
        }
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        body = ""
        try:
            body = (exc.response.text or "")[:300]
        except Exception:
            pass
        return {
            "ok": False,
            "configured": True,
            "env_source": source,
            "model": model_name(),
            "base_url": base_url(),
            "http_status": status,
            "message": body or str(exc)[:300],
        }
    except Exception as exc:
        return {
            "ok": False,
            "configured": True,
            "env_source": source,
            "model": model_name(),
            "base_url": base_url(),
            "message": str(exc)[:500],
        }
