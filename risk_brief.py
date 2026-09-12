from __future__ import annotations
import pandas as pd

PATH_CN={"WS":"长期缺水压力","DR":"历史干旱","SV":"季节性供水波动"}

def _missing(v):
    return v is None or (isinstance(v,float) and pd.isna(v))

def _fmt(v,n=4):
    if _missing(v): return "不可计算"
    try: return f"{float(v):.{n}f}"
    except Exception: return "不可计算"

def baseline_brief(result:dict, material:str|None=None)->str:
    if not result or result.get("data") is None:
        return "目前还没有足够的信息形成完整的风险结果。请补充或核对采购来源、地区和采购占比后继续。"
    s=result["data"]["summary"]; n=result["data"]["nodes"]
    if isinstance(s,list): s=pd.DataFrame(s)
    if isinstance(n,list): n=pd.DataFrame(n)
    if material and not s.empty and material in set(s.material):
        s=s[s.material==material]; n=n[n.material==material]
    paras=[]
    for _,r in s.iterrows():
        mat=str(r.get("material") or "该原材料"); ent=str(r.get("enterprise") or "")
        gn=n[n.material==r.get("material")] if not n.empty and "material" in n.columns else pd.DataFrame()
        topc=gn.sort_values("C",ascending=False).iloc[0] if len(gn) and "C" in gn.columns else None
        paths={"WS":float(r.get("path_contrib_WS") or 0),"DR":float(r.get("path_contrib_DR") or 0),"SV":float(r.get("path_contrib_SV") or 0)}
        total=sum(paths.values()); dom=max(paths,key=paths.get) if total>0 else None
        proc=float(r.get("Coverage") or 0); scored=r.get("scored_coverage"); cov=proc if _missing(scored) else float(scored)
        lines=[f"### {(ent+' · ') if ent else ''}{mat} 当前风险概览",
               f"- **综合风险指数**：{_fmt(r.get('PRWI'))}；已识别采购覆盖 {proc*100:.1f}%，可完成风险评估的采购覆盖 {cov*100:.1f}%。"]
        if dom:
            lines.append(f"- **主要风险来源**：{PATH_CN[dom]}，约占当前综合风险的 {paths[dom]/total*100:.2f}%。三类来源合计为 100%。")
        if topc is not None:
            lines.append(f"- **对企业影响最大的供应地**：{topc.get('node_name')}（{topc.get('node_id')}），约占当前组合风险贡献的 {float(topc.get('contribution_share') or 0)*100:.1f}%。")
        lines.append(f"- **数据可信程度**：{str(r.get('overall_confidence') or 'Unknown')}。风险大小和数据可信程度需要分开判断。")
        identity=result.get("data_identity") or {}
        if identity.get("has_proxy") or identity.get("has_assumption"):
            lines.append("- **数据身份提醒**：本次结果含代理数据或假设参数；这并不等于所有输入都已经真实核验。")
        lines.append("- **建议**：优先核实对企业影响最大的供应地、真实采购占比和低置信度数据；必要时准备监测、库存或替代采购方案。")
        paras.append("\n".join(lines))
    return "\n\n".join(paras)

def scenario_brief(result:dict)->str:
    if not result or result.get("data") is None:
        return "这项比较目前还缺少必要信息，暂时不能形成可靠结论。"
    d=result["data"]; typ=d.get("scenario_type")
    if typ in ["PeakSeason","ExtremeDrought"]:
        delta=d.get("PRWI_delta"); direction="上升" if (delta or 0)>0 else ("下降" if (delta or 0)<0 else "基本不变")
        title="关键用水期压力升高" if typ=="PeakSeason" else "严重干旱再次发生"
        text=(f"### {title}\n- 当前综合风险指数为 {_fmt(d.get('PRWI_baseline'))}，在这种情况下为 {_fmt(d.get('PRWI_scenario'))}，变化 {_fmt(delta)}，整体风险{direction}。\n"
              "- 这是一项压力比较，用于识别脆弱点，并不表示这种情况一定会发生。\n- **建议**：同时关注具体供应地、供应缺口和数据可信程度，不要只看一个总分。")
    elif typ=="AqueductFuture":
        text=(f"### 未来水环境发生变化\n- 在 {d.get('year')} / {d.get('path')} 的设定下，当前综合风险指数为 {_fmt(d.get('PRWI_baseline'))}，变化后为 {_fmt(d.get('PRWI_future'))}，变化 {_fmt(d.get('PRWI_delta'))}。\n"
              "- 这类结果只用于比较趋势，不能当成确定预测。\n- **建议**：把结果用于识别需要优先监测和准备替代方案的供应地。")
    elif typ=="NodeFailure":
        text=(f"### 主要供应节点中断\n- 受影响供应地：{d.get('target_node_name')}（{d.get('target_node_id')}），本次设定影响比例为 {float(d.get('failure_fraction_f') or 0)*100:.0f}%。\n"
              f"- 预计供应损失 {_fmt(d.get('gross_loss'))}，库存可缓冲 {_fmt(d.get('inventory_used'))}，替代供应 {_fmt(d.get('replacement_allocated'))}，仍未满足的需求 {_fmt(d.get('unmet_demand'))}。\n"
              f"- 剩余供应的综合风险为 {_fmt(d.get('conditional_PRWI'))}；采购集中度 {_fmt(d.get('procurement_hhi'))}，风险集中度 {_fmt(d.get('risk_hhi'))}。\n"
              "- **建议**：总风险数字下降不一定代表更安全，还要同时看供应缺口、替代能力和采购是否变得更集中。")
    else:
        text="不同情况下的比较已经完成，可以继续查看对供应和管理决策的影响。"
    if result.get("warnings"): text += "\n- **需要注意**："+"；".join(result["warnings"])
    return text

def data_audit_brief(validation:dict)->str:
    if not validation: return "还没有完成资料检查。"
    out=["### 资料完整性检查"]; cov=validation.get("coverage",{})
    if isinstance(cov,dict) and cov: out.append("- 已识别采购信息："+"；".join(f"{k} {float(v)*100:.1f}%" for k,v in cov.items()))
    for title,key in [("需要你确认","issues"),("提示","warnings"),("还缺的信息","data_gaps")]:
        vals=validation.get(key,[])
        if vals: out.append(f"- **{title}**："+"；".join(map(str,vals)))
    out.append("- 未识别的采购份额不会被当成 0，也不会自动把已知部分放大到 100%。")
    return "\n".join(out)
