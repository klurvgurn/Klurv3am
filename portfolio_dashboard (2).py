"""
PortfolioIQ Pro  v5.0  — white/daytime theme, real pipeline data
Run:  streamlit run portfolio_dashboard.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Paths — patched by Colab cell ─────────────────────────────────────────────
PIPELINE_OUT = os.environ.get("PIPELINE_OUT", os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(PIPELINE_OUT, "cleaned_sp500_stocks_since_2020.csv")
RANKED_CSV= os.path.join(PIPELINE_OUT, "all_strategies_ranked.csv")
ML_LB_CSV = os.path.join(PIPELINE_OUT, "ml_model_leaderboard.csv")
ML_DET_CSV= os.path.join(PIPELINE_OUT, "ml_model_detail.csv")
DESC_CSV  = os.path.join(PIPELINE_OUT, "descriptive_stats.csv")
EF_CSV    = os.path.join(PIPELINE_OUT, "efficient_frontier_points.csv")

RISK_FREE = 0.0525
TX_BPS    = 5

SECTOR_MAP = {
    "Information Technology":["NVDA","AVGO","AAPL","MSFT","AMD","AMAT","ADI","ADBE","ADSK","AKAM","APH","CDNS","FTNT","GOOG","GOOGL","HPE","IBM","INTC","IT","KEYS","KLAC","LRCX","MCHP","MPWR","MU","NOW","NTAP","NXPI","ORCL","PAYC","PAYX","QCOM","ROP","SNPS","STX","TDY","TEL","TER","TXN","VRSN","WDC","ZBRA"],
    "Health Care":["LLY","MCK","ABT","ABBV","A","ALGN","BDX","BIIB","BMY","BSX","CAH","CI","CNC","CVS","DHR","DVA","ELV","GILD","HCA","HOLX","HUM","IDXX","IQV","ISRG","JNJ","MDT","MRK","PFE","REGN","RMD","STE","SYK","TMO","UHS","UNH","VRTX","WAT","ZBH","ZTS"],
    "Industrials":["FIX","PWR","EME","HWM","GE","HON","MMM","CAT","DE","ETN","ROK","EMR","PH","AME","FAST","GWW","ITW","LMT","NOC","RTX","TDG","TT","CARR","OTIS","FTV","IR","LDOS","LHX","MAS","PCAR","PNR","SNA","SWK","TXT","WAB","XYL"],
    "Financials":["JPM","BAC","WFC","GS","MS","BLK","AXP","CB","MET","PRU","TRV","AFL","AIG","AIZ","AJG","ALL","CINF","CMA","COF","DFS","FITB","HIG","KEY","LNC","MTB","PFG","PGR","PNC","RF","SCHW","STT","USB","ZION"],
    "Consumer Disc":["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","DHI","F","GM","LEN","LOW","NFLX","RL","ROST","YUM","CCL","CMG","EBAY","EXPE","HAS","MAR","MGM","PHM","POOL","TSCO"],
    "Consumer Stapl":["KO","PEP","PG","WMT","COST","PM","MO","CL","GIS","HRL","HSY","K","KHC","KMB","MDLZ","MKC","MNST","SJM","SYY","TAP","TSN","CAG","CPB"],
    "Energy":["XOM","CVX","COP","EOG","MPC","PSX","VLO","BKR","DVN","HAL","HES","KMI","LNG","MRO","OKE","OXY","PXD","SLB","WMB"],
    "Utilities":["AEP","ATO","AEE","AWK","ED","CMS","CNP","DTE","DUK","EIX","ETR","ES","EXC","FE","LNT","NEE","NI","NRG","PEG","PPL","SO","SRE","WEC","XEL"],
    "Real Estate":["AMT","CCI","DLR","EQIX","PLD","PSA","SPG","O","AVB","EQR","ESS","FRT","MAA","REG","UDR","VTR","WELL"],
    "Materials":["LIN","APD","SHW","ECL","DD","DOW","FCX","IP","MOS","NEM","NUE","PKG","PPG","VMC"],
    "Comm. Services":["META","GOOGL","NFLX","DIS","T","VZ","CMCSA","CHTR","EA","IPG","LYV","OMC","TMUS","TTWO"],
}
ALL_SECTORS = sorted(SECTOR_MAP.keys())
ALL_TICKERS = sorted({t for v in SECTOR_MAP.values() for t in v})

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PortfolioIQ Pro", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── WHITE / DAYTIME CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[data-testid="stApp"],.stApp{background:#F6F8FA!important;color:#1C2128!important;font-family:'Inter',sans-serif!important}
.main .block-container{padding-top:1rem;max-width:1500px}
[data-testid="stSidebar"]{background:#FFFFFF!important;border-right:1px solid #D0D7DE}
[data-testid="stSidebar"] *{color:#1C2128!important}
[data-testid="stSidebar"] label{color:#57606A!important;font-size:0.78rem!important}
.kpi-card{background:#FFFFFF;border:1px solid #D0D7DE;border-radius:8px;padding:16px 20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.kpi-card:hover{border-color:#0969DA;box-shadow:0 2px 8px rgba(9,105,218,0.12)}
.kpi-label{color:#57606A;font-size:0.70rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
.kpi-value{font-size:1.65rem;font-weight:700;font-family:'JetBrains Mono',monospace}
.kpi-sub{color:#57606A;font-size:.72rem;margin-top:2px}
.kpi-green{color:#1A7F37}.kpi-red{color:#CF222E}.kpi-blue{color:#0969DA}.kpi-yellow{color:#9A6700}.kpi-purple{color:#8250DF}
.sec-hdr{font-size:.78rem;font-weight:700;color:#57606A;letter-spacing:.12em;text-transform:uppercase;border-left:3px solid #0969DA;padding-left:10px;margin:20px 0 10px 0}
[data-baseweb="tab-list"]{background:#FFFFFF!important;border-bottom:1px solid #D0D7DE}
[data-baseweb="tab"]{color:#57606A!important;font-size:.82rem!important;font-weight:600!important}
[aria-selected="true"][data-baseweb="tab"]{color:#0969DA!important;border-bottom:2px solid #0969DA!important}
.stButton>button{background:#FFFFFF!important;border:1px solid #D0D7DE!important;color:#1C2128!important;font-weight:600!important;border-radius:6px!important}
.stButton>button:hover{background:#F3F4F6!important;border-color:#0969DA!important}
hr{border-color:#D0D7DE!important}
</style>
""", unsafe_allow_html=True)

# ── Plotly white theme helper (NO xaxis/yaxis in dict to avoid duplicate kwarg) ─
def pl(**extra):
    base = dict(
        paper_bgcolor="white", plot_bgcolor="#F6F8FA",
        font=dict(family="Inter,sans-serif", color="#1C2128", size=11),
        margin=dict(l=50,r=30,t=45,b=40),
        legend=dict(bgcolor="white",bordercolor="#D0D7DE",borderwidth=1,font=dict(size=10)),
    )
    base.update(extra)
    return base

def ax():
    return dict(showgrid=True,gridcolor="#EAEEF2",zeroline=False,
                showline=True,linecolor="#D0D7DE",tickfont=dict(color="#57606A"))

OPT_COLORS={"Max Sharpe":"#CF222E","Min Variance":"#0969DA","Black-Litterman":"#8250DF",
            "Risk Parity":"#1A7F37","Equal Weight":"#9A6700","Mean-CVaR":"#0550AE"}
UNIV_COLORS={"High Sharpe":"#0969DA","PCA Cluster":"#1A7F37","Low Volatility":"#8250DF"}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_prices():
    """Load all 486 tickers from cleaned pipeline CSV.
    Format: Date column + ticker price columns.
    Filename in pipeline: cleaned_sp500_stocks_since_2020.csv
    """
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date")
        df.sort_index(inplace=True)
        df = df.ffill().bfill()
        df = df.dropna(axis=1, how="all")
        return df
    st.error(
        f"Price CSV not found at: {DATA_CSV}\n\n"
        "In Colab, re-run the path patch cell pointing to:\n"
        "cleaned_sp500_stocks_since_2020.csv in your latest/ folder."
    )
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_ranked():
    if not os.path.exists(RANKED_CSV): return None
    df=pd.read_csv(RANKED_CSV)
    if "Rank" not in df.columns:
        df=df.sort_values("Sharpe",ascending=False).reset_index(drop=True)
        df["Rank"]=df.index+1
    return df

@st.cache_data(show_spinner=False)
def load_ml_lb(): return pd.read_csv(ML_LB_CSV) if os.path.exists(ML_LB_CSV) else None

@st.cache_data(show_spinner=False)
def load_ml_det(): return pd.read_csv(ML_DET_CSV) if os.path.exists(ML_DET_CSV) else None

@st.cache_data(show_spinner=False)
def load_desc(): return pd.read_csv(DESC_CSV,index_col=0) if os.path.exists(DESC_CSV) else None

# ── Metric helpers ─────────────────────────────────────────────────────────────
def log_ret(px): return np.log(px/px.shift(1)).dropna()
def ann_r(r):    return (1+r).prod()**(252/max(len(r),1))-1
def ann_v(r):    return r.std()*np.sqrt(252)
def sr(r):       av=ann_v(r); return (ann_r(r)-RISK_FREE)/av if av>0 else np.nan
def mdd(r):      c=(1+r).cumprod(); return (c/c.cummax()-1).min()
def calmar_r(r): m=mdd(r); return ann_r(r)/abs(m) if m!=0 else np.nan
def sortino_r(r):
    ar=ann_r(r); dd=r[r<0].std()*np.sqrt(252)
    return (ar-RISK_FREE)/dd if dd>0 else np.nan

def full_metrics(r, bench=None):
    m=dict(ret=ann_r(r),vol=ann_v(r),sharpe=sr(r),mdd=mdd(r),
           calmar=calmar_r(r),sortino=sortino_r(r),cum=(1+r).prod()-1,
           beta=float("nan"),alpha=float("nan"))
    if bench is not None:
        a=pd.concat([r,bench],axis=1).dropna(); a.columns=["p","b"]
        if len(a)>5 and a["b"].var()>0:
            m["beta"]=np.cov(a["p"],a["b"])[0,1]/a["b"].var()
            m["alpha"]=m["ret"]-(RISK_FREE+m["beta"]*(ann_r(bench)-RISK_FREE))
    return m

# ── Optimisers ─────────────────────────────────────────────────────────────────
def optimise(lr_df, method="Max Sharpe", wcap=0.30):
    tks=list(lr_df.columns); n=len(tks); ew=dict(zip(tks,[1/n]*n))
    if n<2: return ew
    try:
        from pypfopt import EfficientFrontier, expected_returns, BlackLittermanModel
        from pypfopt.risk_models import CovarianceShrinkage
        px=np.exp(lr_df.cumsum())
        mu=expected_returns.mean_historical_return(px)
        cov=CovarianceShrinkage(px).ledoit_wolf()
        bds=(0,wcap)
        if method=="Max Sharpe":
            ef=EfficientFrontier(mu,cov,weight_bounds=bds); ef.max_sharpe(risk_free_rate=RISK_FREE); return dict(ef.clean_weights())
        if method=="Min Variance":
            ef=EfficientFrontier(mu,cov,weight_bounds=bds); ef.min_volatility(); return dict(ef.clean_weights())
        if method=="Black-Litterman":
            mw=np.array([1/n]*n)
            bl=BlackLittermanModel(cov,pi=2.5*cov.values@mw,
               absolute_views=pd.Series(0.10,index=tks),market_caps=dict(zip(tks,mw)))
            blmu=bl.bl_returns()
            ef=EfficientFrontier(blmu,cov,weight_bounds=bds)
            try: ef.max_sharpe(risk_free_rate=RISK_FREE)
            except: ef=EfficientFrontier(blmu,cov,weight_bounds=bds); ef.min_volatility()
            return dict(ef.clean_weights())
        if method=="Risk Parity":
            cv=cov.values
            def obj(w): pv=w@cv@w; rc=w*(cv@w)/pv; return np.sum((rc-1/n)**2)
            res=minimize(obj,np.ones(n)/n,method="SLSQP",bounds=[(0,wcap)]*n,
                        constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
                        options={"maxiter":1000,"ftol":1e-12})
            if res.success:
                w=np.clip(res.x,0,None); w/=w.sum(); return dict(zip(tks,w))
        if method=="Equal Weight": return ew
        if method=="Mean-CVaR":
            def cvar(w): pr=lr_df.values@w; v=np.percentile(pr,5); return -np.mean(pr[pr<=v])
            res=minimize(cvar,np.ones(n)/n,method="SLSQP",bounds=[(0,wcap)]*n,
                        constraints=[{"type":"eq","fun":lambda w:w.sum()-1}])
            if res.success:
                w=np.clip(res.x,0,None); w/=w.sum(); return dict(zip(tks,w))
    except Exception: pass
    return ew

def backtest(px_df, wdict, tx=TX_BPS):
    tks=[t for t in wdict if t in px_df.columns]
    if not tks: return pd.Series(dtype=float), pd.Series(dtype=float)
    r=log_ret(px_df[tks]).dropna()
    w=np.array([wdict.get(t,0) for t in tks]); w=w/w.sum() if w.sum()>0 else np.ones(len(tks))/len(tks)
    pr=r.dot(w)
    if len(pr)>0: pr.iloc[0]-=tx/10_000
    return pr, (1+pr).cumprod()

def mc_frontier(lr_df, n=2000, wcap=0.30):
    mu=lr_df.mean()*252; cov=lr_df.cov()*252; na=len(lr_df.columns)
    rng=np.random.default_rng(42); vols,rets,srs=[],[],[]
    for _ in range(n):
        w=rng.dirichlet(np.ones(na)); w=np.clip(w,0,wcap); w/=w.sum()
        rv=w@mu.values; vv=np.sqrt(w@cov.values@w)
        vols.append(vv); rets.append(rv); srs.append((rv-RISK_FREE)/vv if vv>0 else 0)
    return np.array(vols),np.array(rets),np.array(srs)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading pipeline data…"):
    price_data = load_prices()
    pipe_res   = load_ranked()
    ml_lb      = load_ml_lb()
    ml_det     = load_ml_det()
    desc       = load_desc()

has_px   = not price_data.empty
has_pipe = pipe_res is not None
has_ml   = ml_lb is not None
avail    = sorted(price_data.columns.tolist()) if has_px else ALL_TICKERS

bench_lr  = log_ret(price_data).mean(axis=1) if has_px else pd.Series(dtype=float)
bench_cum = (1+bench_lr).cumprod()           if has_px else pd.Series(dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 PortfolioIQ Pro")
    st.markdown("<small style='color:#57606A'>S&P 500 · 2020–2026 · Blessing James</small>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🏗️ Portfolio Builder")

    mode=st.radio("Selection method",["By Sector","Manual Tickers","Pipeline Universe"],index=0)
    custom=[]

    if mode=="By Sector":
        sels=st.multiselect("Sectors",ALL_SECTORS,default=["Information Technology","Health Care","Industrials"])
        n_per=st.slider("Top-N per sector (by Sharpe)",1,10,3)
        sect_tks=[t for s in sels for t in SECTOR_MAP.get(s,[]) if t in avail]
        if has_px and sect_tks:
            lr_sub=log_ret(price_data[sect_tks]).dropna()
            sr_rnk=((lr_sub.mean()-RISK_FREE/252)/lr_sub.std()).sort_values(ascending=False)
            for s in sels:
                st_list=[t for t in SECTOR_MAP.get(s,[]) if t in sr_rnk.index]
                custom.extend(sr_rnk[st_list].head(n_per).index.tolist())
            custom=list(dict.fromkeys(custom))
        else:
            custom=sect_tks[:20]

    elif mode=="Manual Tickers":
        defs=[t for t in ["NVDA","AVGO","LLY","MCK","FIX","PWR","EME","HWM","VRT","STX"] if t in avail]
        custom=st.multiselect("Tickers",avail,default=defs)

    else:
        uname=st.selectbox("Universe",["High Sharpe","Low Volatility","PCA Cluster"])
        if has_px:
            lr_all=log_ret(price_data).dropna()
            if uname=="High Sharpe":
                custom=((lr_all.mean()-RISK_FREE/252)/lr_all.std()).sort_values(ascending=False).head(10).index.tolist()
            elif uname=="Low Volatility":
                custom=lr_all.std().sort_values().head(10).index.tolist()
            else:
                X=StandardScaler().fit_transform(lr_all.values).T
                nc=min(10,X.shape[0]-1,X.shape[1]-1)
                proj=PCA(n_components=nc).fit_transform(X)
                km=KMeans(n_clusters=min(10,len(lr_all.columns)),random_state=42,n_init=10).fit(proj[:,:2])
                labels=km.predict(proj[:,:2]); reps=[]
                for c in range(km.n_clusters):
                    idx=np.where(labels==c)[0]
                    if len(idx): reps.append(lr_all.columns[idx[np.linalg.norm(proj[idx,:2]-km.cluster_centers_[c],axis=1).argmin()]])
                custom=reps
        else:
            custom=["NVDA","AVGO","LLY","MCK","FIX","PWR","EME","HWM","VRT","STX"]

    if not custom:
        custom=[t for t in ["NVDA","AVGO","LLY","AAPL","MSFT"] if t in avail]

    st.markdown(f"<small style='color:#0969DA'>**{len(custom)} stocks selected · {len(avail)} available**</small>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Optimisation Settings")
    sel_opts=st.multiselect("Optimisers",list(OPT_COLORS.keys()),default=["Max Sharpe","Min Variance","Black-Litterman","Risk Parity"])
    wcap=st.slider("Max weight per stock (%)",5,50,30,5)/100
    tx_ui=st.slider("Transaction cost (bps)",0,30,TX_BPS,1)
    budget=st.number_input("Investment budget (£)",min_value=1000,max_value=10_000_000,value=10_000,step=1000,format="%d")

    if has_pipe:
        st.markdown("---")
        st.markdown("### 🔬 Pipeline Filter")
        f_univ=st.multiselect("Universe",sorted(pipe_res["Strategy"].unique()),default=sorted(pipe_res["Strategy"].unique()))
        f_opt =st.multiselect("Optimiser",sorted(pipe_res["Optimiser"].unique()),default=sorted(pipe_res["Optimiser"].unique()))
        f_freq=st.multiselect("Frequency",sorted(pipe_res["Frequency"].unique()),default=sorted(pipe_res["Frequency"].unique()))
        top_n =st.slider("Top-N strategies",5,30,10)
    else:
        f_univ=f_opt=f_freq=[]; top_n=10

# ══════════════════════════════════════════════════════════════════════════════
# LIVE OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_opt(tks_t, method, wcap, tx, _n):
    tks=list(tks_t)
    if not has_px or not tks: return {}, pd.Series(dtype=float), pd.Series(dtype=float), {}
    valid=[t for t in tks if t in price_data.columns]
    if len(valid)<2: return {}, pd.Series(dtype=float), pd.Series(dtype=float), {}
    lr_df=log_ret(price_data[valid]).dropna()
    w=optimise(lr_df, method=method, wcap=wcap)
    pr,cum=backtest(price_data[valid], w, tx)
    mets=full_metrics(pr, bench_lr) if len(pr)>10 else {}
    return w, pr, cum, mets

_n=len(price_data) if has_px else 0
opt_res={}
if sel_opts and custom and has_px:
    tks_t=tuple(sorted(custom))
    for m in sel_opts:
        w,pr,cum,mets=run_opt(tks_t,m,wcap,tx_ui,_n)
        opt_res[m]={"w":w,"pr":pr,"cum":cum,"mets":mets}

best_m=max(opt_res,key=lambda m:opt_res[m]["mets"].get("sharpe",-99)) if opt_res else None

# ══════════════════════════════════════════════════════════════════════════════
# HEADER + KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div style="padding:12px 0 4px">
<span style="font-size:2rem;font-weight:800;color:#1C2128">PortfolioIQ</span>
<span style="font-size:2rem;font-weight:300;color:#0969DA"> Pro</span>
<div style="color:#57606A;font-size:.78rem;letter-spacing:.1em;margin-top:2px">
INSTITUTIONAL-GRADE PORTFOLIO OPTIMISATION · MINVAR · MAXSHARPE · BLACK-LITTERMAN · RISK PARITY · MEAN-CVAR
</div></div>""",unsafe_allow_html=True)

if has_px:
    n_obs=len(price_data); n_tks=len(price_data.columns)
    drng=f"{price_data.index.min().date()} → {price_data.index.max().date()}"
    src="Pipeline CSV" if os.path.exists(DATA_CSV) else "yfinance fallback"
    st.markdown(f"""<div style="background:#DAFBE1;border:1px solid #1A7F37;border-radius:6px;padding:8px 16px;color:#1A7F37;font-size:.82rem;margin-bottom:12px">
    ✅ &nbsp;<b>{src}</b> · {n_tks} tickers · {n_obs:,} observations · {drng}
    {'&nbsp;·&nbsp;<b>60-strategy results loaded</b>' if has_pipe else ''}&nbsp;{'·&nbsp;<b>ML leaderboard loaded</b>' if has_ml else ''}
    </div>""",unsafe_allow_html=True)
else:
    st.warning("Price CSV not found. Place the pipeline CSV in the same folder as this script.")

# Per-optimiser KPI cards (like Image 2 in the screenshots)
if opt_res:
    cols_k=st.columns(len(opt_res))
    for col,(m,res) in zip(cols_k,opt_res.items()):
        mm=res["mets"]
        ret_val=mm.get("ret",0); sr_val=mm.get("sharpe",0); vol_val=mm.get("vol",0); mdd_val=mm.get("mdd",0)
        ret_cls="kpi-green" if ret_val>0 else "kpi-red"
        col.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">{m.upper()}</div>
        <div class="kpi-value {ret_cls}">{ret_val:+.1%}</div>
        <div class="kpi-sub">Sharpe {sr_val:.2f} | Vol {vol_val:.1%} | MDD {mdd_val:.1%}</div>
        </div>""",unsafe_allow_html=True)
elif has_pipe and pipe_res is not None:
    br=pipe_res.iloc[0]
    cols_k=st.columns(6)
    kpis=[("Best Sharpe",f"{br.get('Sharpe',0):.3f}","kpi-blue",br.get('Optimiser','')),
          ("Ann. Return",f"{br.get('Ann Return',0):+.1%}","kpi-green",br.get('Strategy','')),
          ("Ann. Vol",f"{br.get('Ann Vol',0):.1%}","kpi-yellow",br.get('Frequency','')),
          ("Max Drawdown",f"{br.get('Max Drawdown',0):.1%}","kpi-red","Worst peak-to-trough"),
          ("Calmar",f"{br.get('Calmar',0):.3f}","kpi-purple","Return / |MDD|"),
          ("Alpha vs EW",f"{br.get('Alpha',0):+.1%}","kpi-blue","vs equal-weighted S&P")]
    for col,(_l,_v,_c,_s) in zip(cols_k,kpis):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{_l}</div><div class="kpi-value {_c}">{_v}</div><div class="kpi-sub">{_s}</div></div>',unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs=st.tabs(["📊 Performance","⚖️ Weights","🎯 Efficient Frontier","🔥 Heatmaps",
              "🤖 ML Forecasts","📉 Risk & Drawdowns","🌍 Universe Analysis",
              "📋 Pipeline Results","💰 Capital Breakdown"])

# ─── TAB 1: Performance ────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec-hdr">Cumulative Portfolio Performance (£1 Invested)</div>',unsafe_allow_html=True)
    if opt_res:
        fig=go.Figure()
        ocl=list(OPT_COLORS.values())
        for i,(m,res) in enumerate(opt_res.items()):
            if len(res["cum"])>0:
                s=res["mets"].get("sharpe",0)
                fig.add_trace(go.Scatter(x=res["cum"].index,y=res["cum"].values,
                    name=f"{m} (Sh={s:.2f})",line=dict(color=ocl[i%len(ocl)],width=2),
                    hovertemplate="%{x|%b %Y}<br><b>£%{y:.2f}</b><extra>"+m+"</extra>"))
        if len(bench_cum)>0:
            fig.add_trace(go.Scatter(x=bench_cum.index,y=bench_cum.values,
                name="S&P 500 EW",line=dict(color="#57606A",width=1.5,dash="dot")))
        fig.update_layout(yaxis_title="Portfolio Value (£)",xaxis_title="Date",height=480,
                          xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig,use_container_width=True)

        st.markdown('<div class="sec-hdr">Metrics Comparison</div>',unsafe_allow_html=True)
        rows=[]
        for m,res in opt_res.items():
            mm=res["mets"]
            rows.append({"Optimiser":m,
                "Ann Ret":f"{mm.get('ret',0):+.2%}","Ann Vol":f"{mm.get('vol',0):.2%}",
                "Sharpe":f"{mm.get('sharpe',0):.3f}","Sortino":f"{mm.get('sortino',0):.3f}",
                "Max DD":f"{mm.get('mdd',0):.2%}","Calmar":f"{mm.get('calmar',0):.3f}",
                "Cum Ret":f"{mm.get('cum',0):+.1%}",
                "Beta":f"{mm.get('beta',float('nan')):.3f}" if not np.isnan(mm.get('beta',float('nan'))) else "—",
                "Alpha":f"{mm.get('alpha',float('nan')):+.2%}" if not np.isnan(mm.get('alpha',float('nan'))) else "—"})
        st.dataframe(pd.DataFrame(rows).set_index("Optimiser"),use_container_width=True)
    elif has_pipe:
        st.info("Select stocks in the sidebar and run optimisation for live results. Showing pipeline top-10 below.")
        st.dataframe(pipe_res.head(10)[["Label","Sharpe","Ann Return","Ann Vol","Max Drawdown","Calmar"]],use_container_width=True)
    else:
        st.info("No data loaded. Check file paths.")

# ─── TAB 2: Weights ────────────────────────────────────────────────────────────
with tabs[1]:
    if opt_res:
        sel_w=st.selectbox("View weights for:",list(opt_res.keys()),key="wsel")
        wd=opt_res[sel_w]["w"]
        if wd:
            wdf=pd.DataFrame({"Ticker":list(wd.keys()),"Weight":list(wd.values())})
            wdf=wdf[wdf["Weight"]>0.001].sort_values("Weight",ascending=False)
            wdf["£ Allocated"]=wdf["Weight"]*budget
            c1,c2=st.columns([1,2])
            with c1:
                st.markdown('<div class="sec-hdr">Allocation</div>',unsafe_allow_html=True)
                st.dataframe(wdf.style.format({"Weight":"{:.2%}","£ Allocated":"£{:,.0f}"}),use_container_width=True,hide_index=True)
            with c2:
                fig_wb=go.Figure(go.Bar(x=wdf["Ticker"],y=wdf["Weight"],
                    marker_color=OPT_COLORS.get(sel_w,"#0969DA"),
                    text=wdf["Weight"].map("{:.1%}".format),textposition="outside"))
                fig_wb.update_layout(title=f"{sel_w} — Portfolio Weights",
                    yaxis_tickformat=".0%",height=400,xaxis=ax(),yaxis=ax(),**pl())
                st.plotly_chart(fig_wb,use_container_width=True)

        all_t=sorted({t for res in opt_res.values() for t in res["w"] if res["w"].get(t,0)>0.001})
        if all_t:
            st.markdown('<div class="sec-hdr">Side-by-Side Weight Comparison</div>',unsafe_allow_html=True)
            fig_wc=go.Figure()
            for i,(m,res) in enumerate(opt_res.items()):
                fig_wc.add_trace(go.Bar(name=m,x=all_t,y=[res["w"].get(t,0) for t in all_t],
                    marker_color=list(OPT_COLORS.values())[i%len(OPT_COLORS)]))
            fig_wc.update_layout(barmode="group",title="Weight Comparison",
                yaxis_tickformat=".0%",height=420,xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_wc,use_container_width=True)
    else:
        st.info("Run optimisation via the sidebar to see weights.")

# ─── TAB 3: Efficient Frontier ─────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="sec-hdr">Empirical Efficient Frontier — Custom Universe</div>',unsafe_allow_html=True)
    if has_px and custom:
        valid_ef=[t for t in custom if t in price_data.columns]
        if len(valid_ef)>=3:
            with st.spinner("Simulating 2,000 random portfolios…"):
                lr_ef=log_ret(price_data[valid_ef]).dropna()
                mcv,mcr,mcsr=mc_frontier(lr_ef,n=2000,wcap=wcap)
            fig_ef=go.Figure()
            fig_ef.add_trace(go.Scatter(x=mcv,y=mcr,mode="markers",
                marker=dict(color=mcsr,colorscale="RdYlGn",size=5,opacity=0.55,
                    colorbar=dict(title="Sharpe",tickfont=dict(color="#57606A"),
                                  titlefont=dict(color="#57606A"))),
                name="Random portfolios",
                hovertemplate="Vol:%{x:.1%}<br>Ret:%{y:.1%}<extra></extra>"))
            mks={"Max Sharpe":"star","Min Variance":"diamond","Black-Litterman":"triangle-up",
                 "Risk Parity":"square","Equal Weight":"circle","Mean-CVaR":"cross"}
            for m,res in opt_res.items():
                if len(res["pr"])>0:
                    av2,ar2,sr2=ann_v(res["pr"]),ann_r(res["pr"]),res["mets"].get("sharpe",0)
                    fig_ef.add_trace(go.Scatter(x=[av2],y=[ar2],mode="markers+text",
                        name=f"{m} (Sh={sr2:.2f})",
                        marker=dict(symbol=mks.get(m,"circle"),size=18,
                            color=OPT_COLORS.get(m,"#333"),line=dict(width=1.5,color="white")),
                        text=[m],textposition="top right",
                        textfont=dict(size=10,color=OPT_COLORS.get(m,"#333"))))
            if len(bench_lr)>0:
                bv2,br2=ann_v(bench_lr),ann_r(bench_lr)
                fig_ef.add_trace(go.Scatter(x=[bv2],y=[br2],mode="markers",name="EW Benchmark",
                    marker=dict(symbol="x",size=16,color="#57606A",line=dict(width=2,color="#57606A"))))
            fig_ef.update_layout(
                title=f"Efficient Frontier — {', '.join(valid_ef[:6])}{'…' if len(valid_ef)>6 else ''}",
                xaxis_title="Annualised Volatility",yaxis_title="Annualised Return",height=560,
                xaxis=dict(**ax(),tickformat=".0%"),
                yaxis=dict(**ax(),tickformat=".0%"),
                **pl())
            st.plotly_chart(fig_ef,use_container_width=True)
        else:
            st.warning("Need at least 3 tickers. Select more stocks in the sidebar.")
    elif has_pipe and os.path.exists(EF_CSV):
        st.dataframe(pd.read_csv(EF_CSV),use_container_width=True,hide_index=True)
    else:
        st.info("Select stocks and run optimisation to generate the frontier.")

# ─── TAB 4: Heatmaps ───────────────────────────────────────────────────────────
with tabs[3]:
    if has_pipe:
        mp=pipe_res[pipe_res["Strategy"].isin(f_univ)&pipe_res["Optimiser"].isin(f_opt)&pipe_res["Frequency"].isin(f_freq)]
        fo_h=["Weekly","Monthly","Quarterly","Yearly","Buy & Hold"]
        c1h,c2h=st.columns(2)
        with c1h:
            st.markdown('<div class="sec-hdr">Sharpe — Optimiser × Frequency</div>',unsafe_allow_html=True)
            pvsh=mp.groupby(["Optimiser","Frequency"])["Sharpe"].max().unstack()
            fo2=[f for f in fo_h if f in pvsh.columns]
            if fo2: pvsh=pvsh[fo2]
            fig_h1=go.Figure(go.Heatmap(z=pvsh.values,x=list(pvsh.columns),y=list(pvsh.index),
                colorscale="RdYlGn",zmid=1.0,text=np.round(pvsh.values,2),texttemplate="%{text:.2f}",
                colorbar=dict(title="Sharpe")))
            fig_h1.update_layout(height=320,xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_h1,use_container_width=True)
        with c2h:
            st.markdown('<div class="sec-hdr">Calmar — Optimiser × Frequency</div>',unsafe_allow_html=True)
            pvcal=mp.groupby(["Optimiser","Frequency"])["Calmar"].max().unstack()
            if fo2: pvcal=pvcal[[f for f in fo2 if f in pvcal.columns]]
            fig_h2=go.Figure(go.Heatmap(z=pvcal.values,x=list(pvcal.columns),y=list(pvcal.index),
                colorscale="YlGn",text=np.round(pvcal.values,2),texttemplate="%{text:.2f}",
                colorbar=dict(title="Calmar")))
            fig_h2.update_layout(height=320,xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_h2,use_container_width=True)

        st.markdown('<div class="sec-hdr">Sharpe Distribution by Universe</div>',unsafe_allow_html=True)
        fig_bx=go.Figure()
        for uv,uc in UNIV_COLORS.items():
            sv=pipe_res[pipe_res["Strategy"]==uv]["Sharpe"].dropna()
            if len(sv)>0:
                fig_bx.add_trace(go.Box(y=sv,name=uv,marker_color=uc,
                    boxpoints="all",jitter=0.35,pointpos=0,opacity=0.85))
        fig_bx.add_hline(y=0.14,line_dash="dash",line_color="#57606A",
            annotation_text="EW Benchmark",annotation_font_color="#57606A")
        fig_bx.update_layout(title="Sharpe Distribution — All 60 Strategies",
            yaxis_title="Sharpe Ratio",height=360,showlegend=False,
            xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_bx,use_container_width=True)

        hsp=mp[mp["Strategy"]=="High Sharpe"]
        if len(hsp)>0 and "Turnover (ann)" in hsp.columns:
            st.markdown('<div class="sec-hdr">Turnover vs Sharpe — High Sharpe Universe</div>',unsafe_allow_html=True)
            fsyms={"Weekly":"circle","Monthly":"square","Quarterly":"triangle-up","Yearly":"diamond","Buy & Hold":"cross"}
            fig_to=go.Figure()
            for opt,oc in OPT_COLORS.items():
                so=hsp[hsp["Optimiser"]==opt]
                for _,ro in so.iterrows():
                    fig_to.add_trace(go.Scatter(x=[ro.get("Turnover (ann)",0)],y=[ro["Sharpe"]],
                        mode="markers",showlegend=False,
                        marker=dict(symbol=fsyms.get(ro["Frequency"],"circle"),color=oc,size=13,
                            opacity=0.85,line=dict(width=1,color="white")),
                        hovertemplate=f"<b>{opt}|{ro['Frequency']}</b><br>TO:%{{x:.0f}}%<br>Sh:%{{y:.3f}}<extra></extra>"))
            fig_to.update_layout(xaxis_title="Ann. Turnover (%)",yaxis_title="Sharpe",
                height=380,xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_to,use_container_width=True)
    elif opt_res:
        fig_hx=go.Figure(go.Bar(x=list(opt_res.keys()),
            y=[r["mets"].get("sharpe",0) for r in opt_res.values()],
            marker_color=[OPT_COLORS.get(m,"#0969DA") for m in opt_res],
            text=[f"{r['mets'].get('sharpe',0):.3f}" for r in opt_res.values()],textposition="outside"))
        fig_hx.update_layout(title="Sharpe by Optimiser (Live)",height=350,xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_hx,use_container_width=True)
    else:
        st.info("Load pipeline results or run optimisation.")

# ─── TAB 5: ML Forecasts ───────────────────────────────────────────────────────
with tabs[4]:
    if has_ml and ml_lb is not None:
        mls=ml_lb.sort_values("Avg_RMSE").reset_index(drop=True)
        c1m,c2m=st.columns(2)
        with c1m:
            st.markdown('<div class="sec-hdr">Average RMSE — lower is better</div>',unsafe_allow_html=True)
            fig_ml1=go.Figure(go.Bar(y=mls["Model"],x=mls["Avg_RMSE"],orientation="h",
                marker_color=["#1A7F37" if i==0 else "#0969DA" for i in range(len(mls))],
                text=mls["Avg_RMSE"].map("{:.5f}".format),textposition="outside"))
            fig_ml1.update_layout(xaxis_autorange="reversed",height=340,
                xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_ml1,use_container_width=True)
        da_col=next((c for c in mls.columns if "dir" in c.lower() or "acc" in c.lower()),None)
        with c2m:
            st.markdown('<div class="sec-hdr">Directional Accuracy — above 50% beats random</div>',unsafe_allow_html=True)
            if da_col:
                fig_ml2=go.Figure(go.Bar(y=mls["Model"],x=mls[da_col],orientation="h",
                    marker_color=["#1A7F37" if d>0.50 else "#CF222E" for d in mls[da_col]],
                    text=mls[da_col].map("{:.2%}".format),textposition="outside"))
                fig_ml2.add_vline(x=0.50,line_dash="dash",line_color="#57606A",
                    annotation_text="Random 50%",annotation_font_color="#57606A")
                fig_ml2.update_layout(xaxis_tickformat=".0%",height=340,
                    xaxis=ax(),yaxis=ax(),**pl())
                st.plotly_chart(fig_ml2,use_container_width=True)

        mls2=mls.copy()
        mls2.insert(0,"Rank",range(1,len(mls2)+1))
        mls2.insert(1,"★",["★ BEST" if i==0 else "" for i in range(len(mls2))])
        st.dataframe(mls2,use_container_width=True,hide_index=True)

        if ml_det is not None:
            st.markdown('<div class="sec-hdr">Per-Ticker Breakdown</div>',unsafe_allow_html=True)
            tk_sel=st.selectbox("Ticker",sorted(ml_det["Ticker"].unique()))
            dsub=ml_det[ml_det["Ticker"]==tk_sel].sort_values("RMSE")
            fig_det=go.Figure(go.Bar(x=dsub["Model"],y=dsub["RMSE"],
                marker_color=["#1A7F37" if i==0 else "#0969DA" for i in range(len(dsub))]))
            fig_det.update_layout(title=f"RMSE by Model — {tk_sel}",height=300,
                xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_det,use_container_width=True)

        st.info("**Key finding (§6.1.2):** ARIMA outperforms all ML alternatives. On noisy daily-return series the bias-variance trade-off favours parsimony — ARIMA (5 parameters) cannot overfit; gradient-boosted models have hundreds.")
    else:
        st.info("Run `sp500_full_pipeline_v3.py` to generate `ml_model_leaderboard.csv`.")

# ─── TAB 6: Risk & Drawdowns ───────────────────────────────────────────────────
with tabs[5]:
    if opt_res:
        st.markdown('<div class="sec-hdr">Rolling 30-day Volatility</div>',unsafe_allow_html=True)
        fig_vol=go.Figure()
        for i,(m,res) in enumerate(opt_res.items()):
            if len(res["pr"])>30:
                rv=res["pr"].rolling(30).std()*np.sqrt(252)
                fig_vol.add_trace(go.Scatter(x=rv.index,y=rv.values,name=m,
                    line=dict(width=1.5,color=list(OPT_COLORS.values())[i%len(OPT_COLORS)])))
        if len(bench_lr)>30:
            bv=bench_lr.rolling(30).std()*np.sqrt(252)
            fig_vol.add_trace(go.Scatter(x=bv.index,y=bv.values,name="EW Benchmark",
                line=dict(color="#57606A",dash="dot",width=1.5)))
        fig_vol.add_hline(y=0.25,line_dash="dash",line_color="#CF222E",
            annotation_text="25% risk cap (§4.5)",annotation_font_color="#CF222E")
        fig_vol.update_layout(yaxis_tickformat=".0%",height=360,
            xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_vol,use_container_width=True)

        st.markdown('<div class="sec-hdr">Drawdown Over Time</div>',unsafe_allow_html=True)
        fig_dd=go.Figure()
        for i,(m,res) in enumerate(opt_res.items()):
            if len(res["pr"])>0:
                c=(1+res["pr"]).cumprod(); dd_s=(c/c.cummax()-1)
                col_dd=list(OPT_COLORS.values())[i%len(OPT_COLORS)]
                fig_dd.add_trace(go.Scatter(x=dd_s.index,y=dd_s.values,name=m,
                    fill="tozeroy",fillcolor=col_dd+"22",
                    line=dict(color=col_dd,width=1.5)))
        fig_dd.update_layout(yaxis_tickformat=".0%",height=360,
            xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_dd,use_container_width=True)

        st.markdown('<div class="sec-hdr">Rolling 1-Year Sharpe Ratio</div>',unsafe_allow_html=True)
        fig_rsr=go.Figure()
        for i,(m,res) in enumerate(opt_res.items()):
            if len(res["pr"])>252:
                rsr=res["pr"].rolling(252).mean()*252/(res["pr"].rolling(252).std()*np.sqrt(252))
                fig_rsr.add_trace(go.Scatter(x=rsr.index,y=rsr.values,name=m,
                    line=dict(width=1.5,color=list(OPT_COLORS.values())[i%len(OPT_COLORS)])))
        fig_rsr.add_hline(y=0,line_dash="dash",line_color="#57606A")
        fig_rsr.update_layout(height=340,xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_rsr,use_container_width=True)
    elif has_px and len(bench_lr)>30:
        bvol=bench_lr.rolling(30).std()*np.sqrt(252)
        fig_bv=go.Figure(go.Scatter(x=bvol.index,y=bvol.values,name="30d Vol",
            line=dict(color="#0969DA",width=1.6),fill="tozeroy",fillcolor="#0969DA22"))
        fig_bv.add_hline(y=0.25,line_dash="dash",line_color="#CF222E",
            annotation_text="25% risk cap",annotation_font_color="#CF222E")
        fig_bv.update_layout(yaxis_tickformat=".0%",height=400,xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_bv,use_container_width=True)
    else:
        st.info("Run optimisation to see risk analysis.")

# ─── TAB 7: Universe Analysis ──────────────────────────────────────────────────
with tabs[6]:
    if has_px and custom:
        vu=[t for t in custom if t in price_data.columns]
        if vu:
            lr_u=log_ret(price_data[vu]).dropna()

            st.markdown('<div class="sec-hdr">Pairwise Correlation Matrix</div>',unsafe_allow_html=True)
            corr_u=lr_u.corr()
            fig_corr=go.Figure(go.Heatmap(z=corr_u.values,x=list(corr_u.columns),y=list(corr_u.index),
                colorscale="RdBu_r",zmid=0,zmin=-1,zmax=1,
                text=np.round(corr_u.values,2),texttemplate="%{text:.2f}",
                colorbar=dict(title="ρ")))
            fig_corr.update_layout(height=max(300,len(vu)*30),xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_corr,use_container_width=True)

            st.markdown('<div class="sec-hdr">Descriptive Statistics</div>',unsafe_allow_html=True)
            u_stats=pd.DataFrame({
                "Ann Ret":lr_u.apply(ann_r).map("{:.2%}".format),
                "Ann Vol":lr_u.apply(ann_v).map("{:.2%}".format),
                "Sharpe":lr_u.apply(sr).map("{:.3f}".format),
                "Skewness":lr_u.skew().map("{:.3f}".format),
                "Kurtosis":lr_u.kurtosis().map("{:.3f}".format),
                "Max DD":lr_u.apply(mdd).map("{:.2%}".format)})
            st.dataframe(u_stats,use_container_width=True)

            st.markdown('<div class="sec-hdr">Individual Stock Cumulative Returns</div>',unsafe_allow_html=True)
            fig_ind=go.Figure()
            cmap_i=px.colors.qualitative.Plotly
            for i,t in enumerate(vu):
                c=(1+lr_u[t]).cumprod()
                fig_ind.add_trace(go.Scatter(x=c.index,y=c.values,name=t,
                    line=dict(width=1.4,color=cmap_i[i%len(cmap_i)])))
            fig_ind.update_layout(yaxis_title="Growth of £1",height=420,
                xaxis=ax(),yaxis=ax(),**pl())
            st.plotly_chart(fig_ind,use_container_width=True)
    elif desc is not None:
        st.dataframe(desc.head(30),use_container_width=True)
    else:
        st.info("Select stocks in the sidebar to see universe analysis.")

# ─── TAB 8: Pipeline Results ───────────────────────────────────────────────────
with tabs[7]:
    if has_pipe:
        fp=pipe_res[pipe_res["Strategy"].isin(f_univ)&pipe_res["Optimiser"].isin(f_opt)&pipe_res["Frequency"].isin(f_freq)].copy()

        st.markdown(f'<div class="sec-hdr">Top {top_n} Strategies</div>',unsafe_allow_html=True)
        top_fp=fp.head(top_n)
        fig_wf=go.Figure(go.Bar(x=top_fp.index+1,y=top_fp["Sharpe"],
            marker_color=[UNIV_COLORS.get(s,"#0969DA") for s in top_fp["Strategy"]],
            text=top_fp["Sharpe"].map("{:.3f}".format),textposition="outside",
            customdata=top_fp.get("Label",top_fp.get("Strategy","")),
            hovertemplate="<b>%{customdata}</b><br>Sharpe:%{y:.3f}<extra></extra>"))
        fig_wf.update_layout(xaxis_title="Rank",yaxis_title="Sharpe",height=360,
            xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_wf,use_container_width=True)

        st.markdown('<div class="sec-hdr">Full 60-Strategy Ranking Waterfall</div>',unsafe_allow_html=True)
        all60=pipe_res.sort_values("Sharpe",ascending=False).reset_index(drop=True); all60["Rank"]=all60.index+1
        fig_60=go.Figure(go.Bar(x=all60["Rank"],y=all60["Sharpe"],width=0.8,
            marker_color=[UNIV_COLORS.get(s,"#0969DA") for s in all60["Strategy"]],
            customdata=all60.get("Label",all60.get("Strategy","")),
            hovertemplate="Rank %{x}<br><b>%{customdata}</b><br>Sharpe:%{y:.3f}<extra></extra>"))
        fig_60.update_layout(xaxis_title="Rank",yaxis_title="Sharpe",height=360,
            xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_60,use_container_width=True)

        st.markdown('<div class="sec-hdr">Sharpe by Frequency — High Sharpe Universe</div>',unsafe_allow_html=True)
        fo_t=["Buy & Hold","Yearly","Monthly","Quarterly","Weekly"]
        hs_r=pipe_res[pipe_res["Strategy"]=="High Sharpe"]
        fig_sf=go.Figure()
        for opt in sorted(hs_r["Optimiser"].unique()):
            s_sf=hs_r[hs_r["Optimiser"]==opt].set_index("Frequency").reindex(fo_t)["Sharpe"]
            fig_sf.add_trace(go.Scatter(x=fo_t,y=s_sf.values,name=opt,mode="lines+markers",
                line=dict(color=OPT_COLORS.get(opt,"#0969DA"),width=2),marker=dict(size=9)))
        fig_sf.update_layout(yaxis_title="Sharpe",height=340,xaxis=ax(),yaxis=ax(),**pl())
        st.plotly_chart(fig_sf,use_container_width=True)

        disp=[c for c in ["Rank","Strategy","Optimiser","Frequency","Sharpe","Ann Return","Ann Vol","Max Drawdown","Calmar","Turnover (ann)"] if c in fp.columns]
        st.dataframe(fp[disp].head(60),use_container_width=True)
    else:
        st.info("Run `sp500_full_pipeline_v3.py` first to generate `all_strategies_ranked.csv`.")

# ─── TAB 9: Capital Breakdown ──────────────────────────────────────────────────
with tabs[8]:
    if opt_res and best_m:
        selcm=st.selectbox("Optimiser",list(opt_res.keys()),key="cap2")
        wc2=opt_res[selcm]["w"]; mc2=opt_res[selcm]["mets"]
        if wc2:
            wn={t:w for t,w in wc2.items() if w>0.001}
            c1c,c2c=st.columns([1,2])
            with c1c:
                st.markdown('<div class="sec-hdr">Portfolio Pie</div>',unsafe_allow_html=True)
                fig_pie=go.Figure(go.Pie(labels=list(wn.keys()),values=list(wn.values()),hole=0.45,
                    marker_colors=px.colors.qualitative.Plotly,textinfo="label+percent",
                    customdata=[v*budget for v in wn.values()],
                    hovertemplate="%{label}<br>Weight:%{value:.2%}<br>£%{customdata:,.0f}<extra></extra>"))
                fig_pie.update_layout(paper_bgcolor="white",font_color="#1C2128",
                    margin=dict(l=10,r=10,t=40,b=10),height=340)
                st.plotly_chart(fig_pie,use_container_width=True)
            with c2c:
                st.markdown('<div class="sec-hdr">Allocation Table</div>',unsafe_allow_html=True)
                atbl=pd.DataFrame([{"Ticker":t,"Weight":f"{w:.2%}","£ Allocated":f"£{w*budget:,.2f}"}
                    for t,w in sorted(wn.items(),key=lambda x:-x[1])])
                st.dataframe(atbl,use_container_width=True,hide_index=True)

            rev_sec={t:s for s,tks in SECTOR_MAP.items() for t in tks}
            sa={}
            for t,w in wn.items(): sa[rev_sec.get(t,"Other")]=sa.get(rev_sec.get(t,"Other"),0)+w
            if sa:
                st.markdown('<div class="sec-hdr">Sector Breakdown</div>',unsafe_allow_html=True)
                sdf=pd.DataFrame(sa.items(),columns=["Sector","Weight"]).sort_values("Weight",ascending=False)
                fig_sec=go.Figure(go.Bar(x=sdf["Sector"],y=sdf["Weight"],
                    marker_color="#0969DA",text=sdf["Weight"].map("{:.1%}".format),textposition="outside"))
                fig_sec.update_layout(yaxis_tickformat=".0%",height=320,xaxis=ax(),yaxis=ax(),**pl())
                st.plotly_chart(fig_sec,use_container_width=True)

            cc1,cc2,cc3,cc4=st.columns(4)
            cc1.markdown(f'<div class="kpi-card"><div class="kpi-label">Proj. Value (1yr)</div><div class="kpi-value kpi-green">£{budget*(1+mc2.get("ret",0)):,.0f}</div><div class="kpi-sub">At {mc2.get("ret",0):+.1%}</div></div>',unsafe_allow_html=True)
            cc2.markdown(f'<div class="kpi-card"><div class="kpi-label">Worst Case (MDD)</div><div class="kpi-value kpi-red">£{budget*(1+mc2.get("mdd",0)):,.0f}</div><div class="kpi-sub">MDD {mc2.get("mdd",0):.1%}</div></div>',unsafe_allow_html=True)
            cc3.markdown(f'<div class="kpi-card"><div class="kpi-label">95% 1-day VaR</div><div class="kpi-value kpi-yellow">£{budget*mc2.get("vol",0.2)/np.sqrt(252)*1.645:,.0f}</div><div class="kpi-sub">Parametric</div></div>',unsafe_allow_html=True)
            cc4.markdown(f'<div class="kpi-card"><div class="kpi-label">Stocks in Portfolio</div><div class="kpi-value kpi-blue">{len(wn)}</div><div class="kpi-sub">of {len(custom)} selected</div></div>',unsafe_allow_html=True)
    else:
        st.info("Run optimisation via the sidebar to see capital breakdown.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#57606A;font-size:.73rem;padding:8px'>PortfolioIQ Pro v5.0 · Blessing James · University of Leicester · May 2026 · All data from pipeline CSV outputs</div>",unsafe_allow_html=True)
