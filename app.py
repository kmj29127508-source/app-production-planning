"""
원예장비 제조업체 총괄생산계획 (APP)
강의록: 스마트제조_06_총괄생산계획 (Chunghun Ha, Hongik Univ.)
Pyomo LP/IP 최적화 + Streamlit 시각화 대시보드
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="원예장비 총괄생산계획",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html,body,[class*="css"]{ font-family:'Noto Sans KR',sans-serif; }
.hdr{
  background:linear-gradient(135deg,#0d3320,#2d7a4f);
  padding:1.4rem 2rem;border-radius:12px;margin-bottom:1.2rem;
}
.hdr h1{color:#fff;margin:0;font-size:1.7rem;font-weight:700;}
.hdr p{color:rgba(255,255,255,.75);margin:.2rem 0 0;font-size:.85rem;}
.kpi{background:linear-gradient(135deg,#f0f7f4,#e4f2eb);border-left:5px solid #2d7a4f;
     border-radius:10px;padding:.8rem 1rem;text-align:center;}
.kpi-lbl{font-size:.68rem;color:#5a7a6a;font-weight:600;text-transform:uppercase;letter-spacing:.5px;}
.kpi-val{font-size:1.25rem;font-weight:700;color:#1a3c2e;font-family:monospace;}
.kpi-unit{font-size:.68rem;color:#5a7a6a;}
.sec{font-size:1rem;font-weight:700;color:#1a3c2e;border-left:4px solid #2d7a4f;
     padding-left:.7rem;margin:1rem 0 .6rem;}
.ok{background:#d1e7dd;border:1px solid #a3cfbb;border-radius:7px;padding:.6rem 1rem;color:#0a3622;font-size:.88rem;}
.warn{background:#fff3cd;border:1px solid #ffc107;border-radius:7px;padding:.6rem 1rem;color:#664d03;font-size:.88rem;}
.fail{background:#f8d7da;border:1px solid #f5c2c7;border-radius:7px;padding:.6rem 1rem;color:#842029;font-size:.88rem;}
.stTabs [data-baseweb="tab-list"]{gap:5px;}
.stTabs [data-baseweb="tab"]{border-radius:6px 6px 0 0;padding:7px 16px;font-weight:500;background:#eef6f1;}
.stTabs [aria-selected="true"]{background:#2d7a4f!important;color:#fff!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 최적화 함수
# ─────────────────────────────────────────────
def run_optimization(demand, W0, I0, I_final,
                     c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C,
                     upw, std_time, ot_limit, model_type):
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Objective, Constraint,
            NonNegativeReals, NonNegativeIntegers,
            SolverFactory, minimize, value
        )
    except Exception as e:
        return None, f"Pyomo 임포트 실패: {e}"

    TH   = len(demand)
    TIME = range(0, TH + 1)
    T    = range(1, TH + 1)

    tv = NonNegativeIntegers if model_type == "IP" else NonNegativeReals

    try:
        m = ConcreteModel()
        m.W = Var(TIME, domain=tv, bounds=(0, None))
        m.H = Var(TIME, domain=tv, bounds=(0, None))
        m.L = Var(TIME, domain=tv, bounds=(0, None))
        m.P = Var(TIME, domain=tv, bounds=(0, None))
        m.I = Var(TIME, domain=tv, bounds=(0, None))
        m.S = Var(TIME, domain=tv, bounds=(0, None))
        m.C = Var(TIME, domain=tv, bounds=(0, None))
        m.O = Var(TIME, domain=tv, bounds=(0, None))

        m.Cost = Objective(
            expr=sum(
                c_W*m.W[t] + c_O*m.O[t] + c_H*m.H[t] + c_L*m.L[t]
                + c_I*m.I[t] + c_S*m.S[t] + c_P*m.P[t] + c_C*m.C[t]
                for t in T
            ),
            sense=minimize
        )

        m.labor     = Constraint(T, rule=lambda m,t: m.W[t] == m.W[t-1]+m.H[t]-m.L[t])
        m.capacity  = Constraint(T, rule=lambda m,t: m.P[t] <= upw*m.W[t] + (1/std_time)*m.O[t])
        m.inventory = Constraint(T, rule=lambda m,t:
                                 m.I[t] == m.I[t-1]+m.P[t]+m.C[t]-demand[t-1]-m.S[t-1]+m.S[t])
        m.overtime  = Constraint(T, rule=lambda m,t: m.O[t] <= ot_limit*m.W[t])
        m.W_0       = Constraint(rule=m.W[0] == W0)
        m.I_0       = Constraint(rule=m.I[0] == I0)
        m.S_0       = Constraint(rule=m.S[0] == 0)
        m.last_inv  = Constraint(rule=m.I[TH] >= I_final)
        m.last_s    = Constraint(rule=m.S[TH] == 0)

        solved = False
        err_msg = ""
        for sn in ["glpk", "cbc", "highs"]:
            try:
                slv = SolverFactory(sn)
                if slv.available():
                    result = slv.solve(m, tee=False)
                    solved = True
                    break
            except Exception as e:
                err_msg += f"{sn}: {e}\n"

        if not solved:
            return None, f"솔버 실행 실패:\n{err_msg}"

        months = [f"{i+1}월" for i in range(TH)]
        rows = []
        for t in T:
            Wt = value(m.W[t])
            Ht = value(m.H[t])
            Lt = value(m.L[t])
            Pt = value(m.P[t])
            It = value(m.I[t])
            St = value(m.S[t])
            Ct = value(m.C[t])
            Ot = value(m.O[t])
            rows.append({
                "월":          months[t-1],
                "수요":         demand[t-1],
                "작업자수":     round(Wt, 2),
                "고용":         round(Ht, 2),
                "해고":         round(Lt, 2),
                "생산량":       round(Pt, 2),
                "기말재고":     round(It, 2),
                "부족재고":     round(St, 2),
                "외주량":       round(Ct, 2),
                "초과시간":     round(Ot, 2),
                "정규임금비용": round(c_W*Wt, 1),
                "초과근무비용": round(c_O*Ot, 1),
                "고용비용":     round(c_H*Ht, 1),
                "해고비용":     round(c_L*Lt, 1),
                "재고비용":     round(c_I*It, 1),
                "부족재고비용": round(c_S*St, 1),
                "재료비":       round(c_P*Pt, 1),
                "하청비용":     round(c_C*Ct, 1),
            })

        df = pd.DataFrame(rows)
        df["총비용"] = (df["정규임금비용"] + df["초과근무비용"] + df["고용비용"]
                       + df["해고비용"] + df["재고비용"] + df["부족재고비용"]
                       + df["재료비"] + df["하청비용"])
        return df, round(value(m.Cost), 2)

    except Exception as e:
        return None, f"모델 오류:\n{traceback.format_exc()}"


# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <h1>🌿 원예장비 제조업체 총괄생산계획 (APP)</h1>
  <p>Aggregate Production Planning · Pyomo LP/IP 최적화 · 스마트제조_06 강의록 (Chunghun Ha, Hongik Univ.)</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 파라미터 설정")

    st.markdown("### 📅 월별 수요 (개/월)")
    n_months = st.selectbox("계획 기간 (월)", [6, 8, 10, 12], index=0)
    preset = {
        6:  [1600, 3000, 3200, 3800, 2200, 2200],
        8:  [1600, 3000, 3200, 3800, 2200, 2200, 2500, 2800],
        10: [1600, 3000, 3200, 3800, 2200, 2200, 2500, 2800, 3100, 2900],
        12: [1600, 3000, 3200, 3800, 2200, 2200, 2500, 2800, 3100, 2900, 2000, 1800],
    }
    demand_list = []
    c2 = st.columns(2)
    for i in range(n_months):
        with c2[i % 2]:
            v = st.number_input(f"{i+1}월", 0, 99999, preset[n_months][i], 100, key=f"d{i}")
            demand_list.append(v)

    st.markdown("### 📦 초기/최종 조건")
    W0      = st.number_input("초기 종업원 수 W₀ (명)", 1, 500, 80, 5)
    I0      = st.number_input("초기 재고 I₀ (개)",      0, 99999, 1000, 100)
    I_final = st.number_input("최종 재고 최솟값 (개)",   0, 99999, 500, 100)

    st.markdown("### 💰 비용 계수 (천원) — 강의록 기본값")
    c_W = st.number_input("c_W 정규임금 (천원/인/월)", 0.0, 9999.0, 640.0, 10.0)
    c_O = st.number_input("c_O 초과근무 (천원/Hr)",    0.0, 999.0,    6.0,  0.5)
    c_H = st.number_input("c_H 고용비용 (천원/인)",    0.0, 9999.0, 300.0, 10.0)
    c_L = st.number_input("c_L 해고비용 (천원/인)",    0.0, 9999.0, 500.0, 10.0)
    c_I = st.number_input("c_I 재고유지 (천원/개/월)", 0.0, 999.0,    2.0,  0.5)
    c_S = st.number_input("c_S 부족재고 (천원/개/월)", 0.0, 999.0,    5.0,  0.5)
    c_P = st.number_input("c_P 재료비   (천원/개)",    0.0, 999.0,   10.0,  1.0)
    c_C = st.number_input("c_C 하청비용  (천원/개)",   0.0, 999.0,   30.0,  1.0)

    st.markdown("### 🏭 작업 파라미터")
    work_days  = st.number_input("작업일수 (일/월)",        1, 31, 20, 1)
    work_hours = st.number_input("작업시간 (시간/일)",        1, 24,  8, 1)
    ot_limit   = st.number_input("초과시간 한도 (Hr/인/월)", 0, 100, 10, 1)
    std_time   = st.number_input("표준작업시간 (시간/개)",  0.1, 20.0, 4.0, 0.5, format="%.1f")
    upw = work_days * work_hours / std_time

    st.markdown("### 🎯 모델 유형")
    mt_label = st.radio("변수 유형", ["LP (연속형)", "IP (정수형)"], index=0)
    mt_code  = "LP" if "LP" in mt_label else "IP"

    st.markdown("---")
    run_btn = st.button("🚀 최적화 실행", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 최적화 실행
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("⏳ Pyomo 최적화 수행 중..."):
        df_res, tc_res = run_optimization(
            demand=demand_list, W0=W0, I0=I0, I_final=I_final,
            c_W=c_W, c_O=c_O, c_H=c_H, c_L=c_L,
            c_I=c_I, c_S=c_S, c_P=c_P, c_C=c_C,
            upw=upw, std_time=std_time, ot_limit=ot_limit,
            model_type=mt_code,
        )
    if df_res is not None:
        st.session_state["df"] = df_res
        st.session_state["tc"] = tc_res
        st.session_state["meta"] = {
            "n": n_months, "mt": mt_code, "W0": W0, "I0": I0, 
            "I_final": I_final, "upw": upw, "std_time": std_time, "ot_limit": ot_limit
        }
    if df_res is None:
        st.error(f"❌ 최적화 실패:\n{tc_res}")
        st.stop()
    st.session_state["df"]   = df_res
    st.session_state["tc"]   = tc_res
    st.session_state["meta"] = dict(
        n=n_months, mt=mt_code, W0=W0, I0=I0, I_final=I_final,
        upw=upw, std_time=std_time, ot_limit=ot_limit,
    )

# 결과 없으면 안내
if "df" not in st.session_state:
    st.info("👈 왼쪽 사이드바에서 파라미터를 설정한 후 **🚀 최적화 실행** 버튼을 눌러주세요.")
    st.markdown("""
    ### 📐 강의록 모델 수식
    **목적함수:**
    ```
    Z = Σ [ 640·W_t + 6·O_t + 300·H_t + 500·L_t
          +   2·I_t + 5·S_t +  10·P_t +  30·C_t ]  (단위: 천원)
    ```
    **제약조건:**
    ```
    ① W_t = W_{t-1} + H_t - L_t         (노동력 균형)
    ② P_t ≤ 40·W_t + 0.25·O_t           (생산능력)
    ③ I_t = I_{t-1}+P_t+C_t-D_{t-1}-S_{t-1}+S_t  (재고균형)
    ④ O_t ≤ 10·W_t                       (초과근무 한도)
    ⑤ W₀=80, I₀=1000, S₀=0              (초기조건)
    ⑥ I₆≥500, S₆=0                      (최종조건)
    ```
    """)
    st.stop()

# ─────────────────────────────────────────────
# 결과 표시
# ─────────────────────────────────────────────
df  = st.session_state["df"]
tc  = st.session_state["tc"]
M   = st.session_state["meta"]
ml  = df["월"].tolist()

st.success(f"✅ 최적화 완료! ({M['mt']})  최소 비용 = **{tc:,.1f} 천원** = {tc/1000:,.3f} 백만원")

# KPI
st.markdown('<div class="sec">📊 핵심 성과 지표</div>', unsafe_allow_html=True)
total_demand  = int(df["수요"].sum())
total_prod    = df["생산량"].sum()
total_out     = df["외주량"].sum()
total_backlog = df["부족재고"].sum()
avg_inv       = df["기말재고"].mean()
final_inv     = df["기말재고"].iloc[-1]
svc           = max(0.0, (1 - total_backlog / max(total_demand, 1)) * 100)

kpis = [
    ("최소 총비용",  f"{tc:,.0f}",        "천원"),
    ("총 수요",      f"{total_demand:,}",  "개"),
    ("정규 생산",    f"{total_prod:,.0f}", "개"),
    ("외주 생산",    f"{total_out:,.0f}",  "개"),
    ("평균 재고",    f"{avg_inv:,.0f}",    "개"),
    ("최종 재고",    f"{final_inv:,.0f}",  "개"),
    ("총 부족재고",  f"{total_backlog:,.0f}", "개"),
    ("서비스율",     f"{svc:.1f}",         "%"),
]
cols = st.columns(8)
for (lbl, val, unit), col in zip(kpis, cols):
    with col:
        st.markdown(f"""<div class="kpi">
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")
a1, a2 = st.columns(2)
with a1:
    if total_backlog == 0:
        st.markdown('<div class="ok">✅ 부족재고 없음 — 수요 완전 충족</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="warn">⚠️ 부족재고 {total_backlog:,.0f}개 (서비스율 {svc:.1f}%)</div>', unsafe_allow_html=True)
with a2:
    if final_inv >= M["I_final"]:
        st.markdown(f'<div class="ok">✅ 최종 재고 {final_inv:,.0f}개 ≥ 목표 {M["I_final"]:,}개</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fail">❌ 최종 재고 {final_inv:,.0f}개 &lt; 목표 {M["I_final"]:,}개</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
BG  = "rgba(240,247,244,0.5)"
FNT = "Noto Sans KR"

def lay(h=400, leg=False):
    d = dict(height=h, plot_bgcolor=BG, paper_bgcolor="white",
             font=dict(family=FNT),
             yaxis=dict(gridcolor="rgba(200,220,210,0.5)"),
             margin=dict(t=35, b=20))
    if leg:
        d["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    return d

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏭 생산계획 개요", "👷 인력 계획", "📦 재고 분석",
    "💰 비용 분석", "🔍 제약조건 검증", "📋 상세 결과표",
    "🆚 전략 비교", "💡 계획 평가 및 권고",
])

# ── TAB 1: 생산계획 개요 ──────────────────────
with tab1:
    st.markdown('<div class="sec">월별 생산량 vs 수요</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ml, y=df["생산량"], name="정규 생산",
                         marker_color="#2d7a4f", opacity=.85,
                         text=df["생산량"].round(0).astype(int), textposition="inside"))
    if df["외주량"].sum() > 0:
        fig.add_trace(go.Bar(x=ml, y=df["외주량"], name="외주 생산",
                             marker_color="#e76f51", opacity=.85,
                             text=df["외주량"].round(0).astype(int), textposition="inside"))
    if df["부족재고"].sum() > 0:
        fig.add_trace(go.Bar(x=ml, y=df["부족재고"], name="부족재고",
                             marker_color="#e63946", opacity=.6))
    fig.add_trace(go.Scatter(x=ml, y=df["수요"], name="수요",
                             mode="lines+markers+text",
                             line=dict(color="#264653", width=3, dash="dash"),
                             marker=dict(size=10, symbol="diamond"),
                             text=df["수요"].astype(int), textposition="top center",
                             textfont=dict(color="#264653", size=11)))
    fig.update_layout(barmode="stack", yaxis_title="수량 (개)", **lay(430, True))
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="sec">생산 구성 비율</div>', unsafe_allow_html=True)
        pd_d = {k: v for k, v in {"정규생산": df["생산량"].sum(), "외주생산": df["외주량"].sum()}.items() if v > 0}
        fig2 = go.Figure(go.Pie(
            labels=list(pd_d.keys()), values=[round(v,1) for v in pd_d.values()],
            marker_colors=["#2d7a4f","#e76f51"], hole=.5,
            textinfo="label+percent+value", textfont_size=12))
        fig2.update_layout(height=300, margin=dict(t=10,b=10), paper_bgcolor="white", font=dict(family=FNT))
        st.plotly_chart(fig2, use_container_width=True)
    with cr:
        st.markdown('<div class="sec">생산-수요 갭</div>', unsafe_allow_html=True)
        gap = (df["생산량"] + df["외주량"]) - df["수요"]
        fig3 = go.Figure(go.Bar(
            x=ml, y=gap,
            marker_color=["#2d7a4f" if v>=0 else "#e63946" for v in gap],
            text=[f"{v:+.0f}" for v in gap], textposition="outside"))
        fig3.add_hline(y=0, line_color="#264653", line_width=1.5)
        fig3.update_layout(yaxis_title="갭 (개)", **lay(300))
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: 인력 계획 ────────────────────────
with tab2:
    st.markdown('<div class="sec">월별 인력 현황</div>', unsafe_allow_html=True)
    fig_wf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_wf.add_trace(go.Bar(x=ml, y=df["고용"], name="고용", marker_color="#52b788", opacity=.85,
                            text=df["고용"].round(1), textposition="outside"), secondary_y=False)
    fig_wf.add_trace(go.Bar(x=ml, y=-df["해고"], name="해고(음수)", marker_color="#e63946", opacity=.85,
                            text=df["해고"].round(1), textposition="outside"), secondary_y=False)
    fig_wf.add_trace(go.Scatter(x=ml, y=df["작업자수"], name="작업자 수",
                                mode="lines+markers+text", line=dict(color="#264653", width=3),
                                marker=dict(size=10), text=df["작업자수"].round(1),
                                textposition="top center", textfont=dict(color="#264653", size=11)),
                     secondary_y=True)
    fig_wf.update_layout(height=420, barmode="relative", plot_bgcolor=BG, paper_bgcolor="white",
                         font=dict(family=FNT), legend=dict(orientation="h", y=1.12), margin=dict(t=40,b=20))
    fig_wf.update_yaxes(title_text="고용/해고 (명)", secondary_y=False)
    fig_wf.update_yaxes(title_text="작업자 수 (명)", secondary_y=True)
    st.plotly_chart(fig_wf, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sec">초과근무 vs 한도</div>', unsafe_allow_html=True)
        max_ot = [M["ot_limit"]*w for w in df["작업자수"]]
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Bar(x=ml, y=df["초과시간"], name="실제 초과근무",
                                marker_color="#f4a261", opacity=.85,
                                text=df["초과시간"].round(1), textposition="outside"))
        fig_ot.add_trace(go.Scatter(x=ml, y=max_ot, name="최대 한도",
                                    mode="lines+markers", line=dict(color="#e63946", dash="dot", width=2)))
        fig_ot.update_layout(yaxis_title="초과시간 (Hr/월)", legend=dict(orientation="h",y=1.1), **lay(310))
        st.plotly_chart(fig_ot, use_container_width=True)
    with cb:
        st.markdown('<div class="sec">생산 가동률</div>', unsafe_allow_html=True)
        max_cap = [M["upw"]*w + o/M["std_time"] for w,o in zip(df["작업자수"], df["초과시간"])]
        util = [(p/mc*100) if mc>0 else 0 for p,mc in zip(df["생산량"], max_cap)]
        fig_u = go.Figure(go.Bar(x=ml, y=util,
                                 marker_color=["#e63946" if u>95 else "#f4a261" if u>80 else "#2d7a4f" for u in util],
                                 text=[f"{u:.1f}%" for u in util], textposition="outside"))
        fig_u.add_hline(y=100, line_dash="dash", line_color="#e63946", annotation_text="100% 한계", line_width=2)
        fig_u.add_hline(y=85,  line_dash="dot",  line_color="#f4a261", annotation_text="85% 권장",  line_width=1.5)
        fig_u.update_layout(yaxis_title="가동률 (%)", yaxis_range=[0,115], **lay(310))
        st.plotly_chart(fig_u, use_container_width=True)

# ── TAB 3: 재고 분석 ────────────────────────
with tab3:
    st.markdown('<div class="sec">월별 재고 추이</div>', unsafe_allow_html=True)
    fig_inv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_inv.add_trace(go.Scatter(x=ml, y=df["기말재고"], name="기말 재고",
                                 mode="lines+markers+text", fill="tozeroy",
                                 fillcolor="rgba(69,123,157,0.2)",
                                 line=dict(color="#457b9d", width=2.5), marker=dict(size=9),
                                 text=df["기말재고"].round(0).astype(int),
                                 textposition="top center", textfont=dict(size=10)), secondary_y=False)
    if df["부족재고"].sum() > 0:
        fig_inv.add_trace(go.Bar(x=ml, y=df["부족재고"], name="부족재고",
                                 marker_color="#e63946", opacity=.7,
                                 text=df["부족재고"].round(0).astype(int), textposition="outside"),
                          secondary_y=True)
    fig_inv.add_hline(y=M["I_final"], line_dash="dash", line_color="#2d7a4f",
                      annotation_text=f"목표 {M['I_final']:,}개", line_width=2)
    fig_inv.update_layout(height=430, plot_bgcolor=BG, paper_bgcolor="white",
                          font=dict(family=FNT), legend=dict(orientation="h",y=1.1), margin=dict(t=40,b=20))
    fig_inv.update_yaxes(title_text="재고량 (개)", secondary_y=False)
    fig_inv.update_yaxes(title_text="부족재고 (개)", secondary_y=True)
    st.plotly_chart(fig_inv, use_container_width=True)

    ca2, cb2 = st.columns(2)
    with ca2:
        st.markdown('<div class="sec">재고 변동 Waterfall</div>', unsafe_allow_html=True)
        wv  = [M["I0"]] + df["기말재고"].tolist()
        wl  = ["초기"] + ml
        dlt = [wv[0]] + [wv[i+1]-wv[i] for i in range(len(wv)-1)]
        fig_wf2 = go.Figure(go.Waterfall(
            x=wl, y=dlt, measure=["absolute"]+["relative"]*M["n"],
            increasing={"marker":{"color":"#2d7a4f"}},
            decreasing={"marker":{"color":"#e63946"}},
            connector={"line":{"color":"rgba(0,0,0,.3)"}}))
        fig_wf2.update_layout(yaxis_title="재고 변동 (개)", **lay(310))
        st.plotly_chart(fig_wf2, use_container_width=True)
    with cb2:
        st.markdown('<div class="sec">재고 회전율</div>', unsafe_allow_html=True)
        turnover = [d/max(i,1) for d,i in zip(df["수요"], df["기말재고"])]
        fig_tr = go.Figure(go.Bar(x=ml, y=turnover,
                                  marker_color=["#e63946" if t>5 else "#f4a261" if t>2 else "#2d7a4f" for t in turnover],
                                  text=[f"{t:.2f}" for t in turnover], textposition="outside"))
        fig_tr.add_hline(y=2.0, line_dash="dot", line_color="#457b9d", annotation_text="권장 2.0", line_width=1.5)
        fig_tr.update_layout(yaxis_title="재고 회전율", **lay(310))
        st.plotly_chart(fig_tr, use_container_width=True)

# ── TAB 4: 비용 분석 ────────────────────────
with tab4:
    cost_cols = ["정규임금비용","초과근무비용","고용비용","해고비용",
                 "재고비용","부족재고비용","재료비","하청비용"]
    cost_clrs = ["#2d7a4f","#f4a261","#52b788","#e63946",
                 "#457b9d","#c77dff","#06d6a0","#e76f51"]

    st.markdown('<div class="sec">월별 비용 구성 (천원)</div>', unsafe_allow_html=True)
    fig_c = go.Figure()
    for cc, cl_c in zip(cost_cols, cost_clrs):
        if df[cc].sum() > 0:
            fig_c.add_trace(go.Bar(x=ml, y=df[cc], name=cc, marker_color=cl_c, opacity=.85))
    fig_c.add_trace(go.Scatter(x=ml, y=df["총비용"], name="총비용",
                               mode="lines+markers", line=dict(color="#264653",width=3), marker=dict(size=9)))
    fig_c.update_layout(barmode="stack", yaxis_title="비용 (천원)", **lay(430, True))
    st.plotly_chart(fig_c, use_container_width=True)

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown('<div class="sec">비용 항목 비율</div>', unsafe_allow_html=True)
        cs = {c: df[c].sum() for c in cost_cols if df[c].sum()>0}
        fig_cp = go.Figure(go.Pie(labels=list(cs.keys()), values=[round(v,1) for v in cs.values()],
                                  marker_colors=cost_clrs[:len(cs)], hole=.45,
                                  textinfo="label+percent", textfont_size=11))
        fig_cp.update_layout(height=320, margin=dict(t=10,b=10), paper_bgcolor="white", font=dict(family=FNT))
        st.plotly_chart(fig_cp, use_container_width=True)
    with cd2:
        st.markdown('<div class="sec">누적 비용 추이</div>', unsafe_allow_html=True)
        cum = df["총비용"].cumsum()
        fig_cum = go.Figure(go.Scatter(x=ml, y=cum, mode="lines+markers",
                                       fill="tozeroy", fillcolor="rgba(45,122,79,.15)",
                                       line=dict(color="#2d7a4f",width=3), marker=dict(size=9),
                                       text=[f"{v:,.0f}" for v in cum],
                                       textposition="top center", textfont=dict(size=10)))
        fig_cum.update_layout(yaxis_title="누적 비용 (천원)", **lay(320))
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown('<div class="sec">비용 항목 요약표</div>', unsafe_allow_html=True)
    tbl = pd.DataFrame({
        "비용 항목":       cost_cols,
        "연간 합계 (천원)": [round(df[c].sum(),1) for c in cost_cols],
        "비율 (%)":       [round(df[c].sum()/max(tc,1)*100,1) for c in cost_cols],
        "월 평균 (천원)":  [round(df[c].mean(),1) for c in cost_cols],
    })
    st.dataframe(tbl.style.background_gradient(subset=["연간 합계 (천원)"], cmap="Greens"),
                 use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div style="text-align:right;font-family:monospace;font-size:1.1rem;color:#1a3c2e;
                font-weight:700;padding:.5rem 1rem;background:#e8f5ee;border-radius:8px;margin-top:.5rem;">
        🏆 최소 총비용: {tc:,.1f} 천원 = {tc/1000:,.3f} 백만원
    </div>""", unsafe_allow_html=True)

# ── TAB 5: 제약조건 검증 ─────────────────────
with tab5:
    st.markdown('<div class="sec">제약조건 충족 검증</div>', unsafe_allow_html=True)
    checks = []
    for i, row in df.iterrows():
        t   = i + 1
        Wp  = M["W0"] if t==1 else df.loc[i-1,"작업자수"]
        Ip  = M["I0"] if t==1 else df.loc[i-1,"기말재고"]
        Sp  = 0.0     if t==1 else df.loc[i-1,"부족재고"]
        mc  = M["upw"]*row["작업자수"] + row["초과시간"]/M["std_time"]
        eW  = Wp + row["고용"] - row["해고"]
        eI  = Ip + row["생산량"] + row["외주량"] - row["수요"] - Sp + row["부족재고"]
        for cname, lhs, rhs, op in [
            (f"① 노동력 W_{t}", row["작업자수"], eW, "eq"),
            (f"② 생산능력 P_{t}", row["생산량"], mc, "le"),
            (f"③ 재고균형 I_{t}", row["기말재고"], eI, "eq"),
            (f"④ 초과근무 O_{t}", row["초과시간"], M["ot_limit"]*row["작업자수"], "le"),
        ]:
            ok = abs(lhs-rhs)<0.5 if op=="eq" else lhs<=rhs+0.5
            checks.append({"월":row["월"],"제약":cname,"좌변":round(lhs,2),"우변":round(rhs,2),"충족":"✅" if ok else "❌"})

    df_chk = pd.DataFrame(checks)
    pass_n = (df_chk["충족"]=="✅").sum()
    cv1, cv2 = st.columns([1,3])
    with cv1:
        pct = pass_n/max(len(df_chk),1)*100
        st.metric("제약조건 충족률", f"{pct:.1f}%", f"{pass_n}/{len(df_chk)}")
        cls = "ok" if pct==100 else "fail"
        msg = "✅ 모든 제약 충족" if pct==100 else "❌ 미충족 제약 존재"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
    with cv2:
        st.dataframe(df_chk.style.apply(
            lambda r: ["background:#d1e7dd" if v=="✅" else "background:#f8d7da" for v in r],
            subset=["충족"]), use_container_width=True, hide_index=True, height=380)

    # 레이더 차트
    st.markdown('<div class="sec">🎯 계획 적절성 종합 평가</div>', unsafe_allow_html=True)
    avg_dm = sum(demand_list)/max(M["n"],1)
    s1 = min(svc, 100)
    s2 = min(100, max(0, 100-abs(avg_inv-avg_dm*0.3)/max(avg_dm*0.3,1)*100))
    s3 = max(0, min(100, 100-(df["고용"].sum()+df["해고"].sum())/max(M["W0"],1)*10))
    s4 = max(0, min(100, 100-(tc/max(sum(demand_list),1))/max(c_P,1)*10))
    s5 = min(100, final_inv/max(M["I_final"],1)*100) if M["I_final"]>0 else 100
    mc2 = [M["upw"]*w+o/M["std_time"] for w,o in zip(df["작업자수"],df["초과시간"])]
    s6  = float(np.mean([(p/mc*100) if mc>0 else 0 for p,mc in zip(df["생산량"],mc2)]))
    cats   = ["서비스율","재고 적절성","인력 안정성","비용 효율성","최종재고 달성","가동률"]
    scores = [s1,s2,s3,s4,s5,s6]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(r=scores+[scores[0]], theta=cats+[cats[0]],
                                    fill="toself", fillcolor="rgba(45,122,79,.2)",
                                    line=dict(color="#2d7a4f",width=2.5), marker=dict(size=8), name="현재 계획"))
    fig_r.add_trace(go.Scatterpolar(r=[80]*(len(cats)+1), theta=cats+[cats[0]],
                                    mode="lines", line=dict(color="#457b9d",dash="dash",width=1.5), name="목표(80점)"))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100]),
                                   angularaxis=dict(tickfont=dict(family=FNT,size=12))),
                        height=420, paper_bgcolor="white", font=dict(family=FNT),
                        legend=dict(orientation="h",y=-0.15), margin=dict(t=30,b=60))
    rr1, rr2 = st.columns([2,1])
    with rr1:
        st.plotly_chart(fig_r, use_container_width=True)
    with rr2:
        st.markdown("**📊 지표별 점수**")
        for cat, sc in zip(cats, scores):
            color = "#2d7a4f" if sc>=80 else "#f4a261" if sc>=60 else "#e63946"
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #eee;">
              <span>{cat}</span><span style="color:{color};font-weight:700;font-family:monospace;">{sc:.1f}점</span>
            </div>""", unsafe_allow_html=True)
        avg_sc = float(np.mean(scores))
        oc = "#2d7a4f" if avg_sc>=80 else "#f4a261" if avg_sc>=60 else "#e63946"
        st.markdown(f"""<div style="margin-top:1rem;padding:.8rem;background:#f0f7f4;border-radius:8px;text-align:center;">
          <div style="font-size:.82rem;color:#5a7a6a;">종합 점수</div>
          <div style="font-size:2rem;font-weight:700;color:{oc};font-family:monospace;">{avg_sc:.1f}</div>
          <div style="font-size:.72rem;color:#888;">/ 100점</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 6: 상세 결과표 ──────────────────────
with tab6:
    st.markdown('<div class="sec">📋 월별 총괄생산계획 상세 결과</div>', unsafe_allow_html=True)
    
    # 1. 상단 요약 정보 (변수 참조 에러 방지를 위해 M에서 직접 호출)
    st.code(f"""모델: {M['mt']}  |  계획기간: {M['n']}개월  |  최소총비용: {tc:,.1f} 천원
목적함수: Z = Σ[{c_W:.0f}·W + {c_O:.1f}·O + {c_H:.0f}·H + {c_L:.0f}·L + {c_I:.1f}·I + {c_S:.1f}·S + {c_P:.1f}·P + {c_C:.1f}·C]
제약: ①인력균형  ②생산능력  ③재고균형  ④초과근무한도""")

    # 2. disp 변수 정의 (이게 반드시 for문보다 위에 있어야 NameError가 안 납니다)
    disp = df.copy()

    # 3. 합계/평균 행 계산
    summary_values = {}
    for col in disp.columns:
        if col == "월":
            summary_values[col] = "합계/평균"
        elif col == "작업자수":
            summary_values[col] = round(disp[col].mean(), 1)
        elif pd.api.types.is_numeric_dtype(disp[col]):
            summary_values[col] = round(disp[col].sum(), 1)
        else:
            summary_values[col] = ""

    # 4. 데이터 합치기
    summary_df = pd.DataFrame([summary_values])
    disp = pd.concat([disp, summary_df], ignore_index=True)

    # 5. 스타일링 함수 수정 (ValueError 방지: 열의 개수만큼 리스트 반환)
    def hl(s):
        # s는 행(Row) 시리즈입니다. 행 전체에 스타일을 입히려면 열 개수만큼 리스트를 줘야 합니다.
        is_last = (s.name == len(disp) - 1)
        return ["font-weight:bold; background-color:#e8f5ee" if is_last else "" for _ in s]

    # 6. 표 출력
    st.dataframe(
        disp.style.apply(hl, axis=1)
                  .format({c: "{:,.1f}" for c in disp.columns if c != "월"}),
        use_container_width=True, 
        hide_index=True, 
        height=540
    )

    # 7. 다운로드 버튼
    csv_out = disp.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 결과 CSV 다운로드",
        data=csv_out,
        file_name=f"APP_{M['n']}개월_{M['mt']}.csv",
        mime="text/csv",
        type="primary"
    )
# --- 새로 교체할 코드 (650행 이후부터 파일 끝까지) ---
# ── TAB 7: 전략 비교 ─────────────────────────
with tab7:
    if 'df' in st.session_state:
        # 세션에서 최적화 결과 가져오기
        df_opt = st.session_state.df
        tc_opt = st.session_state.tc
        M = st.session_state.meta
        
        # NameError 방지: 사이드바에 입력된 최신 값을 직접 참조하거나 세션에서 가져옴
        show_strategy_comparison(
            df_opt, tc_opt, M, 
            demand_list=demand_list, # 사이드바에서 정의된 리스트
            c_W=c_W, c_O=c_O, c_H=c_H, c_L=c_L, 
            c_I=c_I, c_S=c_S, c_P=c_P, c_C=c_C
        )
    else:
        st.info("🚀 사이드바에서 '최적화 실행' 버튼을 먼저 눌러주세요.")

# ── TAB 8: 계획 평가 및 권고 ─────────────────
with tab8:
    if 'df' in st.session_state:
        df_opt = st.session_state.df
        tc_opt = st.session_state.tc
        M = st.session_state.meta
        
        show_plan_evaluation(
            df_opt, tc_opt, M, 
            demand_list=demand_list,
            c_W=c_W, c_O=c_O, c_H=c_H, c_L=c_L, 
            c_I=c_I, c_S=c_S, c_P=c_P, c_C=c_C
        )
    else:
        st.info("🚀 사이드바에서 '최적화 실행' 버튼을 먼저 눌러주세요.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:.78rem;">🌿 원예장비 제조업체 총괄생산계획 시스템 · Hongik Univ</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 🆕 신규 기능: 전략 비교 + 계획 평가
# ═══════════════════════════════════════════════════════════════



