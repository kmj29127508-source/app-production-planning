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
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Pyomo 가용 여부 체크
# ─────────────────────────────────────────────
try:
    from pyomo.environ import (
        ConcreteModel, Var, Objective, Constraint,
        NonNegativeReals, NonNegativeIntegers,
        SolverFactory, minimize, value
    )
    PYOMO_OK = True
except ImportError:
    PYOMO_OK = False


# ─────────────────────────────────────────────
# 최적화 엔진
# ─────────────────────────────────────────────
def solve_app(demand, W0, I0, I_final,
              c_W, c_O, c_H, c_L, c_I, c_S, c_P, c_C,
              ot_limit, upw, std_time,
              model_type="LP"):
    """
    강의록 수식 그대로:
      Z = Σ [ c_W·W_t + c_O·O_t + c_H·H_t + c_L·L_t
              + c_I·I_t + c_S·S_t + c_P·P_t + c_C·C_t ]
    s.t.
      W_t  = W_{t-1} + H_t - L_t
      P_t  ≤ upw·W_t + (1/std_time)·O_t
      I_t  = I_{t-1} + P_t + C_t - D_{t-1} - S_{t-1} + S_t
      O_t  ≤ ot_limit·W_t
      초기·최종 조건
    """
    if not PYOMO_OK:
        return None, "Pyomo 미설치"

    TH   = len(demand)
    TIME = range(0, TH + 1)
    T    = range(1, TH + 1)

    tv = NonNegativeIntegers if model_type == "IP" else NonNegativeReals

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

    m.labor     = Constraint(T, rule=lambda m, t:
                             m.W[t] == m.W[t-1] + m.H[t] - m.L[t])
    m.capacity  = Constraint(T, rule=lambda m, t:
                             m.P[t] <= upw * m.W[t] + (1/std_time) * m.O[t])
    m.inventory = Constraint(T, rule=lambda m, t:
                             m.I[t] == m.I[t-1] + m.P[t] + m.C[t]
                             - demand[t-1] - m.S[t-1] + m.S[t])
    m.overtime  = Constraint(T, rule=lambda m, t:
                             m.O[t] <= ot_limit * m.W[t])

    m.W_0      = Constraint(rule=m.W[0] == W0)
    m.I_0      = Constraint(rule=m.I[0] == I0)
    m.S_0      = Constraint(rule=m.S[0] == 0)
    m.last_inv = Constraint(rule=m.I[TH] >= I_final)
    m.last_s   = Constraint(rule=m.S[TH] == 0)

    solved = False
    for sn in ["glpk", "cbc", "highs"]:
        try:
            slv = SolverFactory(sn)
            if slv.available():
                slv.solve(m, tee=False)
                solved = True
                break
        except Exception:
            continue

    if not solved:
        return None, "솔버 없음: glpk/cbc/highs 중 하나를 설치하세요."

    months = [f"{i+1}월" for i in range(TH)]
    rows = []
    for t in T:
        W_t = value(m.W[t])
        H_t = value(m.H[t])
        L_t = value(m.L[t])
        P_t = value(m.P[t])
        I_t = value(m.I[t])
        S_t = value(m.S[t])
        C_t = value(m.C[t])
        O_t = value(m.O[t])
        rows.append({
            "월":          months[t-1],
            "수요":         demand[t-1],
            "작업자수":     round(W_t, 2),
            "고용":         round(H_t, 2),
            "해고":         round(L_t, 2),
            "생산량":       round(P_t, 2),
            "기말재고":     round(I_t, 2),
            "부족재고":     round(S_t, 2),
            "외주량":       round(C_t, 2),
            "초과시간":     round(O_t, 2),
            "정규임금비용": round(c_W * W_t, 1),
            "초과근무비용": round(c_O * O_t, 1),
            "고용비용":     round(c_H * H_t, 1),
            "해고비용":     round(c_L * L_t, 1),
            "재고비용":     round(c_I * I_t, 1),
            "부족재고비용": round(c_S * S_t, 1),
            "재료비":       round(c_P * P_t, 1),
            "하청비용":     round(c_C * C_t, 1),
        })

    df = pd.DataFrame(rows)
    df["총비용"] = (df["정규임금비용"] + df["초과근무비용"] + df["고용비용"]
                   + df["해고비용"] + df["재고비용"] + df["부족재고비용"]
                   + df["재료비"] + df["하청비용"])
    return df, round(value(m.Cost), 2)


# ─────────────────────────────────────────────
# 페이지 설정 & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="원예장비 총괄생산계획",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{ font-family:'Noto Sans KR',sans-serif; }

.hdr{
  background:linear-gradient(135deg,#0d3320 0%,#1a5c38 50%,#2d7a4f 100%);
  padding:1.4rem 2rem; border-radius:12px; margin-bottom:1.2rem;
}
.hdr h1{ color:#fff; margin:0; font-size:1.75rem; font-weight:700; }
.hdr p { color:rgba(255,255,255,.72); margin:.25rem 0 0; font-size:.85rem; }

.kpi{
  background:linear-gradient(135deg,#f0f7f4,#e4f2eb);
  border-left:5px solid #2d7a4f; border-radius:10px;
  padding:.8rem 1rem; text-align:center;
  box-shadow:0 2px 8px rgba(45,122,79,.12);
}
.kpi-lbl{ font-size:.7rem; color:#5a7a6a; font-weight:600;
          text-transform:uppercase; letter-spacing:.6px; }
.kpi-val{ font-size:1.3rem; font-weight:700; color:#1a3c2e;
          font-family:'IBM Plex Mono',monospace; }
.kpi-unit{ font-size:.7rem; color:#5a7a6a; }

.sec{ font-size:1rem; font-weight:700; color:#1a3c2e;
      border-left:4px solid #2d7a4f; padding-left:.7rem;
      margin:1.1rem 0 .6rem; }

.fbox{
  background:#f8f9fa; border:1px solid #dee2e6;
  border-left:4px solid #2d7a4f; border-radius:6px;
  padding:.9rem; font-family:'IBM Plex Mono',monospace;
  font-size:.8rem; color:#333; white-space:pre-wrap; margin-bottom:.8rem;
}
.ok  { background:#d1e7dd;border:1px solid #a3cfbb;border-radius:7px;padding:.65rem 1rem;color:#0a3622;font-size:.88rem; }
.warn{ background:#fff3cd;border:1px solid #ffc107;border-radius:7px;padding:.65rem 1rem;color:#664d03;font-size:.88rem; }
.fail{ background:#f8d7da;border:1px solid #f5c2c7;border-radius:7px;padding:.65rem 1rem;color:#842029;font-size:.88rem; }

.stTabs [data-baseweb="tab-list"]{gap:5px}
.stTabs [data-baseweb="tab"]{border-radius:6px 6px 0 0;padding:7px 16px;
  font-weight:500;background:#eef6f1;}
.stTabs [aria-selected="true"]{background:#2d7a4f!important;color:#fff!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <h1>🌿 원예장비 제조업체 총괄생산계획 (APP)</h1>
  <p>Aggregate Production Planning · Pyomo LP/IP 최적화 · 스마트제조_06 강의록 (Chunghun Ha, Hongik Univ.)</p>
</div>
""", unsafe_allow_html=True)

if not PYOMO_OK:
    st.error("⚠️ Pyomo 미설치. `pip install pyomo` 후 glpk 솔버를 설치하세요.")
    st.code("pip install pyomo\nconda install -c conda-forge glpk  # 또는 apt install glpk-utils")
    st.stop()

# ─────────────────────────────────────────────
# 사이드바 — 파라미터
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 파라미터 설정")

    # ── 계획 기간 & 수요 ────────────────────
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

    # ── 초기/최종 조건 ──────────────────────
    st.markdown("### 📦 초기/최종 조건")
    W0      = st.number_input("초기 종업원 수 W₀ (명)", 1, 500, 80, 5)
    I0      = st.number_input("초기 재고 I₀ (개)",      0, 99999, 1000, 100)
    I_final = st.number_input("최종 재고 최솟값 (개)",   0, 99999, 500, 100)

    # ── 비용 계수 (강의록 기본값 고정 표시 + 편집 가능) ──
    st.markdown("### 💰 비용 계수 (천원)")
    st.caption("강의록 기본값: c_W=640, c_O=6, c_H=300, c_L=500, c_I=2, c_S=5, c_P=10, c_C=30")
    with st.expander("계수 직접 편집 (고급)"):
        c_W = st.number_input("c_W — 정규임금 (천원/인/월)", 0.0, 9999.0, 640.0, 10.0)
        c_O = st.number_input("c_O — 초과근무 (천원/Hr)",    0.0, 999.0,    6.0,  0.5)
        c_H = st.number_input("c_H — 고용비용 (천원/인)",    0.0, 9999.0, 300.0, 10.0)
        c_L = st.number_input("c_L — 해고비용 (천원/인)",    0.0, 9999.0, 500.0, 10.0)
        c_I = st.number_input("c_I — 재고유지 (천원/개/월)", 0.0, 999.0,    2.0,  0.5)
        c_S = st.number_input("c_S — 부족재고 (천원/개/월)", 0.0, 999.0,    5.0,  0.5)
        c_P = st.number_input("c_P — 재료비   (천원/개)",    0.0, 999.0,   10.0,  1.0)
        c_C = st.number_input("c_C — 하청비용  (천원/개)",   0.0, 999.0,   30.0,  1.0)
    if "c_W" not in st.session_state:
        c_W, c_O, c_H, c_L = 640.0, 6.0, 300.0, 500.0
        c_I, c_S, c_P, c_C =   2.0, 5.0,  10.0,  30.0

    # ── 작업 파라미터 ───────────────────────
    st.markdown("### 🏭 작업 파라미터")
    work_days  = st.number_input("작업일수 (일/월)",         1, 31, 20, 1)
    work_hours = st.number_input("작업시간 (시간/일)",         1, 24,  8, 1)
    ot_limit   = st.number_input("초과시간 한도 (Hr/인/월)",  0, 100, 10, 1)
    std_time   = st.number_input("작업표준시간 (시간/개)",   0.1, 20.0, 4.0, 0.5, format="%.1f")

    upw = work_days * work_hours / std_time   # 40 ea/인/월 (기본)

    # ── 모델 유형 ───────────────────────────
    st.markdown("### 🎯 최적화 모델")
    mt_label = st.radio("변수 유형", ["LP (연속형)", "IP (정수형)"], index=0)
    mt_code  = "LP" if "LP" in mt_label else "IP"

    run_btn = st.button("🔄 최적화 실행", type="primary", use_container_width=True)

    # 목적함수 미리보기
    with st.expander("📐 목적함수 확인"):
        st.markdown(f"""<div class="fbox">Z = Σ [
  {c_W:.0f}·W_t   정규임금
  {c_O:.1f}·O_t     초과근무
  {c_H:.0f}·H_t   고용비용
  {c_L:.0f}·L_t   해고비용
  {c_I:.1f}·I_t     재고유지
  {c_S:.1f}·S_t     부족재고
  {c_P:.1f}·P_t    재료비
  {c_C:.1f}·C_t    하청비용 ]
(단위: 천원)</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 최적화 실행
# ─────────────────────────────────────────────
if run_btn:
    with st.spinner("🔍 Pyomo 최적화 수행 중 …"):
        df_res, tc_res = solve_app(
            demand=demand_list, W0=W0, I0=I0, I_final=I_final,
            c_W=c_W, c_O=c_O, c_H=c_H, c_L=c_L,
            c_I=c_I, c_S=c_S, c_P=c_P, c_C=c_C,
            ot_limit=ot_limit, upw=upw, std_time=std_time,
            model_type=mt_code,
        )
    
    if df_res is not None:
        # 결과를 세션 스테이트에 명시적으로 저장
        st.session_state.df = df_res
        st.session_state.tc = tc_res
        st.session_state.meta = dict(
            n=n_months, mt=mt_code, W0=W0, I0=I0, I_final=I_final,
            upw=upw, std_time=std_time, ot_limit=ot_limit,
            c_W=c_W, c_O=c_O, c_H=c_H, c_L=c_L,
            c_I=c_I, c_S=c_S, c_P=c_P, c_C=c_C,
        )
        st.success("✅ 최적화가 완료되었습니다!")
        st.rerun() # 중요: 화면을 즉시 새로고침하여 결과 반영
    else:
        st.error(f"❌ 최적화 실패: {tc_res}")

# 세션에 데이터가 없을 때 안내 문구 추가
if "df" not in st.session_state:
    st.info("👈 왼쪽 사이드바에서 파라미터를 설정한 후 '최적화 실행' 버튼을 눌러주세요.")
    st.stop() # 데이터가 없으면 아래 시각화 코드를 실행하지 않음
    st.success(f"✅ 최적화 완료! ({mt_code})  최소 비용 = **{tc_res:,.1f} 천원** = {tc_res/1000:,.3f} 백만원")

df  = st.session_state.df.copy()
tc  = st.session_state.tc
M   = st.session_state.meta
ml  = df["월"].tolist()

# ─────────────────────────────────────────────
# KPI 카드
# ─────────────────────────────────────────────
st.markdown('<div class="sec">📊 핵심 성과 지표 (KPI)</div>', unsafe_allow_html=True)

total_demand  = sum(df["수요"])
total_prod    = df["생산량"].sum()
total_out     = df["외주량"].sum()
total_backlog = df["부족재고"].sum()
avg_inv       = df["기말재고"].mean()
final_inv     = df["기말재고"].iloc[-1]
svc_level     = max(0.0, (1 - total_backlog / max(total_demand, 1)) * 100)

kpi_info = [
    ("최소 총비용",  f"{tc:,.0f}",        "천원"),
    ("총 수요",      f"{total_demand:,}",  "개"),
    ("정규 생산",    f"{total_prod:,.0f}", "개"),
    ("외주 생산",    f"{total_out:,.0f}",  "개"),
    ("평균 재고",    f"{avg_inv:,.0f}",    "개"),
    ("최종 재고",    f"{final_inv:,.0f}",  "개"),
    ("총 부족재고",  f"{total_backlog:,.0f}", "개"),
    ("서비스율",     f"{svc_level:.1f}",   "%"),
]
kcols = st.columns(8)
for (lbl, val, unit), col in zip(kpi_info, kcols):
    with col:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")
a1, a2 = st.columns(2)
with a1:
    cls = "ok" if total_backlog == 0 else "warn"
    msg = "✅ 부족재고 없음 — 수요 완전 충족" if total_backlog == 0 \
          else f"⚠️ 부족재고 {total_backlog:,.0f}개 (서비스율 {svc_level:.1f}%)"
    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)
with a2:
    if final_inv >= M["I_final"]:
        st.markdown(f'<div class="ok">✅ 최종 재고 {final_inv:,.0f}개 ≥ 목표 {M["I_final"]:,}개</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fail">❌ 최종 재고 {final_inv:,.0f}개 &lt; 목표 {M["I_final"]:,}개</div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 공통 플롯 설정
# ─────────────────────────────────────────────
BG  = "rgba(240,247,244,0.5)"
FNT = "Noto Sans KR"

def lay(h=400, legend_h=False):
    d = dict(height=h, plot_bgcolor=BG, paper_bgcolor="white",
             font=dict(family=FNT),
             yaxis=dict(gridcolor="rgba(200,220,210,0.5)"),
             margin=dict(t=35, b=20))
    if legend_h:
        d["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    return d


# ─────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 생산계획 개요",
    "👷 인력 계획",
    "📦 재고 분석",
    "💰 비용 분석",
    "🔍 제약조건 검증",
    "📋 상세 결과표",
])

# ══════════════════════════════════════
# TAB 1 — 생산계획 개요
# ══════════════════════════════════════
with tab1:
    st.markdown('<div class="sec">월별 생산량 vs 수요</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ml, y=df["생산량"], name="정규 생산",
        marker_color="#2d7a4f", opacity=.85,
        text=df["생산량"].round(0).astype(int), textposition="inside",
    ))
    if df["외주량"].sum() > 0:
        fig.add_trace(go.Bar(
            x=ml, y=df["외주량"], name="외주 생산",
            marker_color="#e76f51", opacity=.85,
            text=df["외주량"].round(0).astype(int), textposition="inside",
        ))
    if df["부족재고"].sum() > 0:
        fig.add_trace(go.Bar(
            x=ml, y=df["부족재고"], name="부족재고",
            marker_color="#e63946", opacity=.6,
        ))
    fig.add_trace(go.Scatter(
        x=ml, y=df["수요"], name="수요",
        mode="lines+markers+text",
        line=dict(color="#264653", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
        text=df["수요"].astype(int), textposition="top center",
        textfont=dict(color="#264653", size=11),
    ))
    fig.update_layout(barmode="stack", yaxis_title="수량 (개)", **lay(430, True))
    st.plotly_chart(fig, use_container_width=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="sec">생산 구성 비율 (연간)</div>', unsafe_allow_html=True)
        pd_dict = {k: v for k, v in
                   {"정규생산": df["생산량"].sum(), "외주생산": df["외주량"].sum()}.items()
                   if v > 0}
        fig_pie = go.Figure(go.Pie(
            labels=list(pd_dict.keys()),
            values=[round(v, 1) for v in pd_dict.values()],
            marker_colors=["#2d7a4f", "#e76f51"],
            hole=.5, textinfo="label+percent+value", textfont_size=12,
        ))
        fig_pie.update_layout(height=300, margin=dict(t=10,b=10),
                              paper_bgcolor="white", font=dict(family=FNT))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cr:
        st.markdown('<div class="sec">생산-수요 갭</div>', unsafe_allow_html=True)
        gap = (df["생산량"] + df["외주량"]) - df["수요"]
        fig_gap = go.Figure(go.Bar(
            x=ml, y=gap,
            marker_color=["#2d7a4f" if v >= 0 else "#e63946" for v in gap],
            text=[f"{v:+.0f}" for v in gap], textposition="outside",
        ))
        fig_gap.add_hline(y=0, line_color="#264653", line_width=1.5)
        fig_gap.update_layout(yaxis_title="갭 (개)", **lay(300))
        st.plotly_chart(fig_gap, use_container_width=True)


# ══════════════════════════════════════
# TAB 2 — 인력 계획
# ══════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">월별 인력 현황 및 변동</div>', unsafe_allow_html=True)

    fig_wf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_wf.add_trace(go.Bar(
        x=ml, y=df["고용"], name="고용",
        marker_color="#52b788", opacity=.85,
        text=df["고용"].round(1), textposition="outside",
    ), secondary_y=False)
    fig_wf.add_trace(go.Bar(
        x=ml, y=-df["해고"], name="해고 (음수 표시)",
        marker_color="#e63946", opacity=.85,
        text=df["해고"].round(1), textposition="outside",
    ), secondary_y=False)
    fig_wf.add_trace(go.Scatter(
        x=ml, y=df["작업자수"], name="총 작업자 수",
        mode="lines+markers+text",
        line=dict(color="#264653", width=3),
        marker=dict(size=10),
        text=df["작업자수"].round(1), textposition="top center",
        textfont=dict(color="#264653", size=11),
    ), secondary_y=True)
    fig_wf.update_layout(
        height=420, barmode="relative", plot_bgcolor=BG,
        paper_bgcolor="white", font=dict(family=FNT),
        legend=dict(orientation="h", y=1.12), margin=dict(t=40,b=20),
    )
    fig_wf.update_yaxes(title_text="고용/해고 (명)", secondary_y=False)
    fig_wf.update_yaxes(title_text="작업자 수 (명)", secondary_y=True)
    st.plotly_chart(fig_wf, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sec">초과근무 시간 vs 한도</div>', unsafe_allow_html=True)
        max_ot = [M["ot_limit"] * w for w in df["작업자수"]]
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Bar(
            x=ml, y=df["초과시간"], name="실제 초과근무",
            marker_color="#f4a261", opacity=.85,
            text=df["초과시간"].round(1), textposition="outside",
        ))
        fig_ot.add_trace(go.Scatter(
            x=ml, y=max_ot, name="최대 한도",
            mode="lines+markers",
            line=dict(color="#e63946", dash="dot", width=2),
            marker=dict(size=7),
        ))
        fig_ot.update_layout(yaxis_title="초과시간 (Hr/월)",
                             legend=dict(orientation="h", y=1.1), **lay(310))
        st.plotly_chart(fig_ot, use_container_width=True)

    with cb:
        st.markdown('<div class="sec">생산 가동률</div>', unsafe_allow_html=True)
        max_cap = [M["upw"] * w + o / M["std_time"]
                   for w, o in zip(df["작업자수"], df["초과시간"])]
        util = [(p / mc * 100) if mc > 0 else 0
                for p, mc in zip(df["생산량"], max_cap)]
        fig_u = go.Figure(go.Bar(
            x=ml, y=util,
            marker_color=["#e63946" if u > 95 else "#f4a261" if u > 80 else "#2d7a4f"
                          for u in util],
            text=[f"{u:.1f}%" for u in util], textposition="outside",
        ))
        fig_u.add_hline(y=100, line_dash="dash", line_color="#e63946",
                        annotation_text="100% 한계", line_width=2)
        fig_u.add_hline(y=85,  line_dash="dot",  line_color="#f4a261",
                        annotation_text="85% 권장", line_width=1.5)
        fig_u.update_layout(yaxis_title="가동률 (%)", yaxis_range=[0, 115], **lay(310))
        st.plotly_chart(fig_u, use_container_width=True)


# ══════════════════════════════════════
# TAB 3 — 재고 분석
# ══════════════════════════════════════
with tab3:
    st.markdown('<div class="sec">월별 재고 및 부족재고 추이</div>', unsafe_allow_html=True)

    fig_inv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_inv.add_trace(go.Scatter(
        x=ml, y=df["기말재고"], name="기말 재고",
        mode="lines+markers+text",
        fill="tozeroy", fillcolor="rgba(69,123,157,0.2)",
        line=dict(color="#457b9d", width=2.5), marker=dict(size=9),
        text=df["기말재고"].round(0).astype(int),
        textposition="top center", textfont=dict(size=10),
    ), secondary_y=False)
    if df["부족재고"].sum() > 0:
        fig_inv.add_trace(go.Bar(
            x=ml, y=df["부족재고"], name="부족재고",
            marker_color="#e63946", opacity=.7,
            text=df["부족재고"].round(0).astype(int), textposition="outside",
        ), secondary_y=True)
    fig_inv.add_hline(y=M["I_final"], line_dash="dash", line_color="#2d7a4f",
                      annotation_text=f"최종 목표 {M['I_final']:,}개", line_width=2)
    fig_inv.update_layout(
        height=430, plot_bgcolor=BG, paper_bgcolor="white", font=dict(family=FNT),
        legend=dict(orientation="h", y=1.1), margin=dict(t=40,b=20),
    )
    fig_inv.update_yaxes(title_text="재고량 (개)", secondary_y=False)
    fig_inv.update_yaxes(title_text="부족재고 (개)", secondary_y=True)
    st.plotly_chart(fig_inv, use_container_width=True)

    ca2, cb2 = st.columns(2)
    with ca2:
        st.markdown('<div class="sec">재고 변동 (Waterfall)</div>', unsafe_allow_html=True)
        wv  = [M["I0"]] + df["기말재고"].tolist()
        wl  = ["초기"] + ml
        dlt = [wv[0]] + [wv[i+1] - wv[i] for i in range(len(wv)-1)]
        fig_wf2 = go.Figure(go.Waterfall(
            x=wl, y=dlt,
            measure=["absolute"] + ["relative"] * M["n"],
            increasing={"marker": {"color": "#2d7a4f"}},
            decreasing={"marker": {"color": "#e63946"}},
            connector={"line": {"color": "rgba(0,0,0,.3)"}},
        ))
        fig_wf2.update_layout(yaxis_title="재고 변동 (개)", **lay(310))
        st.plotly_chart(fig_wf2, use_container_width=True)

    with cb2:
        st.markdown('<div class="sec">재고 회전율 (수요/기말재고)</div>', unsafe_allow_html=True)
        turnover = [d / max(i, 1) for d, i in zip(df["수요"], df["기말재고"])]
        fig_tr = go.Figure(go.Bar(
            x=ml, y=turnover,
            marker_color=["#e63946" if t > 5 else "#f4a261" if t > 2 else "#2d7a4f"
                          for t in turnover],
            text=[f"{t:.2f}" for t in turnover], textposition="outside",
        ))
        fig_tr.add_hline(y=2.0, line_dash="dot", line_color="#457b9d",
                         annotation_text="권장 기준 2.0", line_width=1.5)
        fig_tr.update_layout(yaxis_title="재고 회전율", **lay(310))
        st.plotly_chart(fig_tr, use_container_width=True)


# ══════════════════════════════════════
# TAB 4 — 비용 분석
# ══════════════════════════════════════
with tab4:
    cost_cols = ["정규임금비용","초과근무비용","고용비용","해고비용",
                 "재고비용","부족재고비용","재료비","하청비용"]
    cost_clrs = ["#2d7a4f","#f4a261","#52b788","#e63946",
                 "#457b9d","#c77dff","#06d6a0","#e76f51"]

    st.markdown('<div class="sec">월별 비용 구성 (천원)</div>', unsafe_allow_html=True)
    fig_c = go.Figure()
    for cc, cl_c in zip(cost_cols, cost_clrs):
        if df[cc].sum() > 0:
            fig_c.add_trace(go.Bar(x=ml, y=df[cc], name=cc,
                                   marker_color=cl_c, opacity=.85))
    fig_c.add_trace(go.Scatter(
        x=ml, y=df["총비용"], name="총비용",
        mode="lines+markers",
        line=dict(color="#264653", width=3), marker=dict(size=9),
    ))
    fig_c.update_layout(barmode="stack", yaxis_title="비용 (천원)", **lay(430, True))
    st.plotly_chart(fig_c, use_container_width=True)

    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown('<div class="sec">비용 항목 비율</div>', unsafe_allow_html=True)
        cs = {c: df[c].sum() for c in cost_cols if df[c].sum() > 0}
        fig_cp = go.Figure(go.Pie(
            labels=list(cs.keys()),
            values=[round(v, 1) for v in cs.values()],
            marker_colors=cost_clrs[:len(cs)],
            hole=.45, textinfo="label+percent", textfont_size=11,
        ))
        fig_cp.update_layout(height=320, margin=dict(t=10,b=10),
                             paper_bgcolor="white", font=dict(family=FNT))
        st.plotly_chart(fig_cp, use_container_width=True)

    with cd2:
        st.markdown('<div class="sec">누적 비용 추이</div>', unsafe_allow_html=True)
        cum = df["총비용"].cumsum()
        fig_cum = go.Figure(go.Scatter(
            x=ml, y=cum, mode="lines+markers",
            fill="tozeroy", fillcolor="rgba(45,122,79,.15)",
            line=dict(color="#2d7a4f", width=3), marker=dict(size=9),
            text=[f"{v:,.0f}" for v in cum],
            textposition="top center", textfont=dict(size=10),
        ))
        fig_cum.update_layout(yaxis_title="누적 비용 (천원)", **lay(320))
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown('<div class="sec">비용 항목 요약표</div>', unsafe_allow_html=True)
    tbl = pd.DataFrame({
        "비용 항목":       cost_cols,
        "연간 합계 (천원)": [round(df[c].sum(), 1) for c in cost_cols],
        "비율 (%)":       [round(df[c].sum()/max(tc,1)*100, 1) for c in cost_cols],
        "월 평균 (천원)":  [round(df[c].mean(), 1) for c in cost_cols],
        "최대 (천원)":     [round(df[c].max(), 1) for c in cost_cols],
    })
    st.dataframe(
        tbl.style.background_gradient(subset=["연간 합계 (천원)"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )
    st.markdown(f"""
    <div style="text-align:right;font-family:'IBM Plex Mono',monospace;font-size:1.15rem;
                color:#1a3c2e;font-weight:700;padding:.5rem 1rem;background:#e8f5ee;
                border-radius:8px;margin-top:.5rem;">
        🏆 최소 총비용: {tc:,.1f} 천원 &nbsp;=&nbsp; {tc/1000:,.3f} 백만원
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════
# TAB 5 — 제약조건 검증
# ══════════════════════════════════════
with tab5:
    st.markdown('<div class="sec">제약조건 충족 검증</div>', unsafe_allow_html=True)

    checks = []
    for i, row in df.iterrows():
        t     = i + 1
        W_p   = M["W0"] if t == 1 else df.loc[i-1, "작업자수"]
        I_p   = M["I0"] if t == 1 else df.loc[i-1, "기말재고"]
        S_p   = 0.0     if t == 1 else df.loc[i-1, "부족재고"]
        mc    = M["upw"] * row["작업자수"] + row["초과시간"] / M["std_time"]
        exp_W = W_p + row["고용"] - row["해고"]
        exp_I = I_p + row["생산량"] + row["외주량"] - row["수요"] - S_p + row["부족재고"]

        for cname, lhs, rhs, op in [
            (f"① 노동력 W_{t}=W_{t-1}+H-L",  row["작업자수"], exp_W,         "eq"),
            (f"② 생산능력 P_{t}≤{mc:.1f}",    row["생산량"],  mc,             "le"),
            (f"③ 재고균형 I_{t}=",             row["기말재고"], exp_I,         "eq"),
            (f"④ 초과근무 O_{t}≤{M['ot_limit']}·W", row["초과시간"],
             M["ot_limit"]*row["작업자수"], "le"),
        ]:
            if op == "eq":
                ok = abs(lhs - rhs) < 0.5
            else:
                ok = lhs <= rhs + 0.5
            checks.append({"월": row["월"], "제약": cname,
                           "좌변": round(lhs,2), "우변": round(rhs,2),
                           "충족": "✅" if ok else "❌"})

    df_chk = pd.DataFrame(checks)
    pass_n = (df_chk["충족"] == "✅").sum()

    cv1, cv2 = st.columns([1, 3])
    with cv1:
        pct = pass_n / max(len(df_chk), 1) * 100
        st.metric("제약조건 충족률", f"{pct:.1f}%", f"{pass_n}/{len(df_chk)}")
        if pct == 100:
            st.markdown('<div class="ok">✅ 모든 제약 충족</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="fail">❌ 미충족 제약 존재</div>', unsafe_allow_html=True)
    with cv2:
        st.dataframe(
            df_chk.style.apply(
                lambda r: ["background:#d1e7dd" if v == "✅" else "background:#f8d7da"
                           for v in r], subset=["충족"]),
            use_container_width=True, hide_index=True, height=380,
        )

    # 레이더 차트
    st.markdown('<div class="sec">🎯 계획 적절성 종합 평가</div>', unsafe_allow_html=True)

    avg_dm = sum(demand_list) / max(M["n"], 1)
    s_svc  = min(svc_level, 100)
    s_inv  = min(100, max(0, 100 - abs(avg_inv - avg_dm*0.3)/max(avg_dm*0.3,1)*100))
    s_wf   = max(0, min(100, 100-(df["고용"].sum()+df["해고"].sum())/max(M["W0"],1)*10))
    s_cost = max(0, min(100, 100-(tc/max(sum(demand_list),1))/(max(M["c_P"],1))*10))
    s_finv = min(100, final_inv/max(M["I_final"],1)*100) if M["I_final"]>0 else 100
    max_c2 = [M["upw"]*w+o/M["std_time"] for w,o in zip(df["작업자수"],df["초과시간"])]
    s_util = float(np.mean([(p/mc*100) if mc>0 else 0
                            for p,mc in zip(df["생산량"],max_c2)]))

    cats   = ["서비스율","재고 적절성","인력 안정성","비용 효율성","최종재고 달성","가동률"]
    scores = [s_svc, s_inv, s_wf, s_cost, s_finv, s_util]

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=scores+[scores[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(45,122,79,.2)",
        line=dict(color="#2d7a4f", width=2.5), marker=dict(size=8),
        name="현재 계획",
        hovertemplate="%{theta}: %{r:.1f}점<extra></extra>",
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=[80]*(len(cats)+1), theta=cats+[cats[0]],
        mode="lines", line=dict(color="#457b9d", dash="dash", width=1.5),
        name="목표 기준 (80점)",
    ))
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100]),
            angularaxis=dict(tickfont=dict(family=FNT, size=12)),
        ),
        height=420, paper_bgcolor="white", font=dict(family=FNT),
        legend=dict(orientation="h", y=-0.15), margin=dict(t=30,b=60),
    )
    rr1, rr2 = st.columns([2, 1])
    with rr1:
        st.plotly_chart(fig_r, use_container_width=True)
    with rr2:
        st.markdown("**📊 지표별 점수**")
        for cat, sc in zip(cats, scores):
            color = "#2d7a4f" if sc>=80 else "#f4a261" if sc>=60 else "#e63946"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:1px solid #eee;">
              <span>{cat}</span>
              <span style="color:{color};font-weight:700;
                           font-family:'IBM Plex Mono',monospace;">{sc:.1f}점</span>
            </div>""", unsafe_allow_html=True)
        avg_sc = float(np.mean(scores))
        oc = "#2d7a4f" if avg_sc>=80 else "#f4a261" if avg_sc>=60 else "#e63946"
        st.markdown(f"""
        <div style="margin-top:1rem;padding:.8rem;background:#f0f7f4;
                    border-radius:8px;text-align:center;">
          <div style="font-size:.82rem;color:#5a7a6a;">종합 점수</div>
          <div style="font-size:2rem;font-weight:700;color:{oc};
                      font-family:'IBM Plex Mono',monospace;">{avg_sc:.1f}</div>
          <div style="font-size:.72rem;color:#888;">/ 100점</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════
# TAB 6 — 상세 결과표
# ══════════════════════════════════════
with tab6:
    st.markdown('<div class="sec">📋 월별 총괄생산계획 상세 결과</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="fbox">【강의록 모델】 {M['mt']} 최적화  |  계획기간: {M['n']}개월  |  최소총비용: {tc:,.1f} 천원

목적함수: Z = Σ [{M['c_W']:.0f}·W_t + {M['c_O']:.1f}·O_t + {M['c_H']:.0f}·H_t + {M['c_L']:.0f}·L_t
                + {M['c_I']:.1f}·I_t + {M['c_S']:.1f}·S_t + {M['c_P']:.1f}·P_t + {M['c_C']:.1f}·C_t]

제약조건:
 ① W_t = W_{{t-1}} + H_t - L_t
 ② P_t ≤ {M['upw']:.1f}·W_t + (1/{M['std_time']})·O_t
 ③ I_t = I_{{t-1}} + P_t + C_t - D_{{t-1}} - S_{{t-1}} + S_t
 ④ O_t ≤ {M['ot_limit']}·W_t
 ⑤ W_0={M['W0']}, I_0={M['I0']}, S_0=0
 ⑥ I_{M['n']} ≥ {M['I_final']}, S_{M['n']} = 0</div>""",
                unsafe_allow_html=True)

    disp = df[[
        "월","수요","생산량","외주량","기말재고","부족재고",
        "작업자수","고용","해고","초과시간",
        "정규임금비용","초과근무비용","고용비용","해고비용",
        "재고비용","부족재고비용","재료비","하청비용","총비용",
    ]].copy()

    # 합계/평균 행
    sr = {"월": "합계/평균"}
    for col in disp.columns[1:]:
        sr[col] = round(disp[col].sum(), 1)
    sr["작업자수"] = round(disp["작업자수"].mean(), 1)
    disp = pd.concat([disp, pd.DataFrame([sr])], ignore_index=True)

    def hl(s):
    # n을 len(disp)가 아닌 len(s)로 바꾸어 열의 개수에 맞춥니다.
        n_cols = len(s) 
    # 마지막 행(합계 행)일 경우에만 강조 스타일 적용
        return ["font-weight:bold;background:#e8f5ee"] * n_cols if s.name == len(disp)-1 else [""] * n_cols
    
    st.dataframe(
        disp.style.apply(hl, axis=1)
                  .format({c: "{:,.1f}" for c in disp.columns if c != "월"}),
        use_container_width=True, hide_index=True, height=540,
    )

    csv = disp.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 결과 CSV 다운로드", csv,
        f"APP_결과_{M['n']}개월_{M['mt']}.csv",
        "text/csv", type="primary",
    )

# ─────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.78rem;padding:.8rem;">
🌿 원예장비 제조업체 총괄생산계획 시스템 &nbsp;|&nbsp;
스마트제조_06 강의록 기반 Pyomo LP/IP 최적화 &nbsp;|&nbsp;
Chunghun Ha · Hongik University
</div>""", unsafe_allow_html=True)
