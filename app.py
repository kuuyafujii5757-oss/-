import datetime as dt
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import ndtr

st.set_page_config(page_title='ダイナミックデルタヘッジ・シミュレーター', layout='wide', page_icon='📈')
st.markdown('''<style>
.stApp {background:#050608;color:#ffffff}
.stApp label, .stApp p, .stApp small, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"] {color:#ffffff !important;opacity:1 !important}
/* Streamlit inputs have a pale native background: use dark text only inside them. */
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="select"] [role="combobox"],
.stApp [data-baseweb="select"] [data-testid="stMarkdownContainer"],
.stApp input[type="date"], .stApp input[type="number"],
.stApp input[type="text"] {color:#111111 !important;-webkit-text-fill-color:#111111 !important;}
.stApp [data-baseweb="input"], .stApp [data-baseweb="select"] > div {background:#f0f2f6 !important;border-color:#cbd5e1 !important;}
.stApp [data-baseweb="select"] * {color:#111111 !important;-webkit-text-fill-color:#111111 !important;}
.stApp input::placeholder {color:#555555 !important;-webkit-text-fill-color:#555555 !important;opacity:1}
.stApp input[type="date"] {color-scheme:light;}
.stApp [data-testid="stMetric"] {background:#10151e;border:1px solid #384152;border-radius:9px;padding:12px}
[data-testid="stSidebar"] {background:#10131a}
.stButton button {border:1px solid #3786ff}
</style>''', unsafe_allow_html=True)
st.title('オプション・ダイナミックデルタヘッジ シミュレーター')
st.caption('日本株個別オプション｜条件付きモンテカルロ100経路｜毎営業日終値でリバランス')

SQRT2PI=np.sqrt(2*np.pi)
def bs(S,K,T,r,q,sigma,call=True):
    S=np.asarray(S,dtype=float)
    T=max(float(T),0.)
    if T<=1e-10:
        intrinsic=np.maximum(S-K,0) if call else np.maximum(K-S,0)
        delta=np.where(S>K,1.,np.where(S<K,0.,0.5)) if call else np.where(S<K,-1.,np.where(S>K,0.,-0.5))
        z=np.zeros_like(S)
        return intrinsic,delta,z,z,z
    vol=max(float(sigma),1e-8)
    root=np.sqrt(T)
    d1=(np.log(np.maximum(S,1e-12)/K)+(r-q+0.5*vol**2)*T)/(vol*root)
    d2=d1-vol*root
    n1=ndtr(d1); n2=ndtr(d2); pdf=np.exp(-d1*d1/2)/SQRT2PI
    discq=np.exp(-q*T); discr=np.exp(-r*T)
    if call:
        price=S*discq*n1-K*discr*n2
        delta=discq*n1
        theta=-(S*discq*pdf*vol)/(2*root)-r*K*discr*n2+q*S*discq*n1
    else:
        price=K*discr*ndtr(-d2)-S*discq*ndtr(-d1)
        delta=discq*(n1-1)
        theta=-(S*discq*pdf*vol)/(2*root)+r*K*discr*ndtr(-d2)-q*S*discq*ndtr(-d1)
    gamma=discq*pdf/(S*vol*root)
    vega=S*discq*pdf*root/100 # per 1 volatility percentage point
    return price,delta,gamma,vega,theta/365

def lot_round(x,lot):
    # nearest whole lot, ties away from zero
    return np.sign(x)*np.floor(np.abs(x)/lot+0.5)*lot

@st.cache_data(show_spinner=False, max_entries=12)
def simulate(S0, endpoint, n, expiry_years, K, r, q, iv, forecast_vol, call, signed_contracts, multiplier, initial_shares, initial_stock_price, min_shares, cost, lot, seed):
    rng=np.random.default_rng(seed)
    if n==0:
        paths=np.full((100,1),S0,dtype=float)
    else:
        # Conditional log-price Brownian bridge: endpoint exactly specified.
        z=rng.standard_normal((100,n))
        W=np.concatenate([np.zeros((100,1)),np.cumsum(z,axis=1)],axis=1)
        t=np.arange(n+1)/n
        logpaths=np.log(S0)+(np.log(endpoint/S0))*t[None,:]+forecast_vol*np.sqrt(1/252)* (W-W[:,-1,None]*t[None,:])
        paths=np.exp(logpaths)
    shares=np.full(100,float(initial_shares))
    # Entry mark-to-market: initial shares bought/sold at their entered entry price.
    cash=np.full(100,-float(initial_shares)*initial_stock_price)
    last=shares.copy()
    for day in range(1,n+1):
        remaining=max(expiry_years-day/252,0)
        _,d,_,_,_=bs(paths[:,day],K,remaining,r,q,iv,call)
        target=-signed_contracts*multiplier*d
        threshold=np.maximum(0.25*np.abs(last),min_shares)
        trigger=np.abs(target-last)>=threshold-1e-10
        desired=lot_round(target,lot)
        trade=np.where(trigger,desired-shares,0.)
        cash-=trade*paths[:,day]+np.abs(trade)*cost
        shares+=trade
        last=np.where(trigger,shares,last)
    # Liquidation-value P/L; do not charge hypothetical liquidation cost.
    return float(np.mean(cash+shares*endpoint)), float(np.std(cash+shares*endpoint,ddof=1)), float(np.mean(shares)), paths

with st.container(border=True):
    st.subheader('初期ポジション設定')

with st.expander('評価の前提・注意点（必ず確認）'):
    st.markdown('''- **欧州型ブラック・ショールズ**で価格とグリークスを算出。実際の契約の権利行使方式、乗数、権利落ち・最終売買日は別途確認してください。
- 株価経路は**対数株価のブラウン橋**で、指定日・指定株価に到達する100経路を生成します。日次ショックの予想ボラを使用し、ドリフトは0%です。端点を固定する条件付き分布では、ドリフトの影響は消えます。
- 25%乖離または最低株数の大きい方でヘッジ発動。目標株数を**100株単位**に四捨五入（中間は絶対値で切り上げ）。ヘッジ後もデルタが残る場合があります。
- 現物損益は初期建値からの評価損益に、日次ヘッジ売買キャッシュフローと1株当たり取引コストを反映します。オプション損益は入力建値からの時価評価です。
- 金利・配当利回りはBS評価に使いますが、現物在庫の資金調達費用・配当受払・オプション売買手数料は**現物PLに含みません**。実務のネットPLでは別途必要です。
- 営業日は**土日を除く平日近似**です。日本の祝日・取引所休業日を完全には反映していません。


st.divider()

with st.expander("免責事項 / Disclaimer"):
    st.markdown("""
### 免責事項

本シミュレーターは、株式オプション取引およびダイナミックデルタヘッジ戦略の研究・教育・分析を目的としています。投資助言または特定の金融商品の売買推奨を目的とするものではありません。

表示されるオプション価格、グリークス、損益およびヘッジ履歴は、数理モデルと入力条件に基づく理論値・推計値です。実際の取引成果とは異なる場合があります。

計算結果の正確性、完全性、将来の収益性は保証されません。投資判断は利用者自身の責任で行ってください。

---

### Disclaimer

This simulator is provided for research, educational, and analytical purposes only. It does not constitute investment advice or a recommendation to trade any financial instrument.

Option prices, Greeks, profit and loss figures, and hedging histories are theoretical or estimated results based on mathematical models and user-defined assumptions. Actual trading outcomes may differ materially.

The accuracy, completeness, and future profitability of the results are not guaranteed. Users are solely responsible for their own investment decisions.
""")

- 株価グラフの現物PLは、**各横軸株価を終点とする別々の条件付き100経路の平均**です。スライダーを動かしても同一経路の実現PLが変化するという意味ではありません。
- 100経路の推定にはモンテカルロ誤差があります。決算イベント、IVスマイル、板の流動性、スリッページ等は再現していません。''')
