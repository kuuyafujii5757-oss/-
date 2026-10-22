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
def simulate(S0, endpoint, n, valuation_dates, expiry_date, K, r, q, iv, forecast_vol, call, signed_contracts, multiplier, initial_shares, initial_stock_price, min_shares, cost, lot, seed):
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
        remaining=max(max((expiry_date - valuation_dates[day]).days/365, 0),0)
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

# JPX official holiday list, cached for 24 hours. Fail closed if unavailable.
@st.cache_data(ttl=86400, show_spinner=False)
def jpx_holidays(years):
    import requests
    import re
    from html.parser import HTMLParser

    class CellReader(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth = 0
            self.parts = []
            self.cells = []
        def handle_starttag(self, tag, attrs):
            if tag in ('td', 'th'):
                self.depth += 1
                if self.depth == 1:
                    self.parts = []
        def handle_data(self, data):
            if self.depth:
                self.parts.append(data)
        def handle_endtag(self, tag):
            if tag in ('td', 'th') and self.depth:
                self.depth -= 1
                if self.depth == 0:
                    self.cells.append(''.join(self.parts).strip())

    url = 'https://www.jpx.co.jp/corporate/about-jpx/calendar/index.html'
    response = requests.get(url, timeout=15, headers={'User-Agent':'Mozilla/5.0'})
    response.raise_for_status()
    parser = CellReader()
    parser.feed(response.text)
    holidays = set()
    for cell in parser.cells:
        for match in re.finditer(r'(20\d{2})\s*[/年.-]\s*(\d{1,2})\s*[/月.-]\s*(\d{1,2})\s*日?', cell):
            try:
                d = dt.date(*map(int, match.groups()))
                if d.year in years:
                    holidays.add(d)
            except ValueError:
                continue
    if not holidays or not all(any(d.year == year for d in holidays) for year in years):
        raise ValueError('JPX休業日一覧を取得・検証できませんでした。')
    return holidays

def trading_dates(entry, expiry, holidays):
    return [d.date() for d in pd.date_range(entry, expiry, freq='D')
            if d.weekday() < 5 and d.date() not in holidays]

with st.container(border=True):
    st.subheader('初期ポジション設定')
    a,b,c,d=st.columns(4)
    with a:
        entry=st.date_input('共通建て日',value=dt.date.today())
        kind=st.selectbox('Put or Call',['Call','Put'])
        side=st.selectbox('オプション 売 or 買',['買','売'])
        contracts=st.number_input('オプション枚数',min_value=1,max_value=10000,value=1,step=1)
    with b:
        S0=st.number_input('建て時 原資産価格 (円)',min_value=1.0,max_value=100000.0,value=30000.0,step=100.0)
        K=st.number_input('権利行使価格 (円)',min_value=1.0,max_value=200000.0,value=30000.0,step=100.0)
        iv_pct=st.number_input('IV (%)',min_value=0.01,max_value=500.0,value=30.0,step=1.0)
        expiry=st.date_input('満期日',value=entry+dt.timedelta(days=45),min_value=entry+dt.timedelta(days=1))
    with c:
        premium=st.number_input('オプション建値 (円/株)',min_value=0.0,value=1000.0,step=50.0)
        multiplier=st.number_input('契約乗数 (株/枚)',min_value=1,max_value=100000,value=100,step=100)
        stock_side=st.selectbox('初期現物 売 or 買',['売','買'])
        stock_qty=st.number_input('初期現物株数',min_value=0,max_value=1000000,value=100,step=100)
    with d:
        stock_entry=st.number_input('現物建値 (円)',min_value=0.01,max_value=1000000.0,value=30000.0,step=100.0)
        forecast_pct=st.number_input('予想ボラ (%)',min_value=0.01,max_value=500.0,value=30.0,step=1.0)
        r_pct=st.number_input('無リスク金利 (%)',value=0.5,step=0.1)
        q_pct=st.number_input('配当利回り (%)',value=0.0,step=0.1)
    e,f,g=st.columns(3)
    with e: cost=st.number_input('現物売買コスト (円/株)',min_value=0.0,value=0.0,step=0.1)
    with f: min_shares=st.number_input('最低ヘッジ乖離 (株)',min_value=1,value=100,step=100)
    with g: seed=st.number_input('乱数シード（再現用）',min_value=0,value=42,step=1)

if expiry<=entry:
    st.error('満期日は建て日より後にしてください。'); st.stop()
try:
    holidays=jpx_holidays(tuple(range(entry.year, expiry.year+1)))
except Exception as exc:
    st.error(f'JPX営業日カレンダーの自動取得に失敗しました: {exc}')
    st.stop()
market_dates=trading_dates(entry, expiry, holidays)
if entry not in market_dates or expiry not in market_dates:
    st.error('建て日と満期日はJPXの営業日を指定してください（現時点では非営業日を指定できません）。')
    st.stop()
valuation_dates=market_dates[market_dates.index(entry):]
remaining_days=len(valuation_dates)-1
if remaining_days<1:
    st.error('建て日と満期日の間に評価可能な営業日がありません。')
    st.stop()
signed_contracts=contracts*(1 if side=='買' else -1)
initial_shares=stock_qty*(1 if stock_side=='買' else -1)
call=kind=='Call'; iv=iv_pct/100; fv=forecast_pct/100;r=r_pct/100;q=q_pct/100
# Keep sliders and numeric inputs synchronized across Streamlit reruns.
def sync_day_from_slider():
    st.session_state.eval_day_number = st.session_state.eval_day_slider

def sync_day_from_number():
    st.session_state.eval_day_slider = st.session_state.eval_day_number

def sync_price_from_slider():
    st.session_state.eval_price_number = st.session_state.eval_price_slider

def sync_price_from_number():
    st.session_state.eval_price_slider = st.session_state.eval_price_number

if 'eval_day_number' not in st.session_state:
    st.session_state.eval_day_number = min(remaining_days, 10)
if 'eval_price_number' not in st.session_state:
    st.session_state.eval_price_number = int(np.clip(S0, 1, 100000))

st.session_state.eval_day_number = int(np.clip(st.session_state.eval_day_number, 0, remaining_days))
st.session_state.eval_day_slider = st.session_state.eval_day_number
st.session_state.eval_price_number = int(np.clip(st.session_state.eval_price_number, 1, 100000))
st.session_state.eval_price_slider = st.session_state.eval_price_number

st.divider()
with st.container(border=True):
    st.subheader('評価日・原資産価格')
    x,y=st.columns(2)
    with x:
        st.slider('建て日からの経過営業日（平日ベース）', min_value=0,
                  max_value=remaining_days, step=1, key='eval_day_slider',
                  on_change=sync_day_from_slider)
        day=st.number_input('経過営業日を直接入力', min_value=0,
                            max_value=remaining_days, step=1, key='eval_day_number',
                            on_change=sync_day_from_number)
        st.caption(f'建て日: {entry} / 満期日: {expiry} / 評価日目安: {pd.Timestamp(valuation_dates[day]):%Y-%m-%d}')
    with y:
        min_s=1
        max_s=100000
        st.slider('評価時の株価 (円)', min_value=min_s, max_value=max_s,
                  step=1, key='eval_price_slider', on_change=sync_price_from_slider)
        target=st.number_input('評価時の株価を直接入力 (円)', min_value=min_s,
                               max_value=max_s, step=1, key='eval_price_number',
                               on_change=sync_price_from_number)

# Calendar-year remaining maturity; time progresses in trading days approximated by 252/year.
expiry_T=max((expiry-entry).days/365,1/365)
T=max((expiry-valuation_dates[day]).days/365,0)
price,delta,gamma,vega,theta=bs(target,K,T,r,q,iv,call)
option_pl=float(signed_contracts*multiplier*(price-premium))
if day==0:
    stock_pl=float(initial_shares*(target-stock_entry)); stock_std=0; avg_shares=initial_shares
else:
    stock_pl,stock_std,avg_shares,_=simulate(S0,float(target),day,tuple(valuation_dates[:day+1]),expiry,K,r,q,iv,fv,call,signed_contracts,multiplier,initial_shares,stock_entry,min_shares,cost,100,seed)
net_pl=option_pl+stock_pl
m1,m2,m3,m4=st.columns(4)
m1.metric('オプション損益',f'{option_pl:+,.0f} 円')
m2.metric('現物ヘッジ損益（100経路平均）',f'{stock_pl:+,.0f} 円')
m3.metric('ネット損益',f'{net_pl:+,.0f} 円')
m4.metric('平均ヘッジ残高',f'{avg_shares:+,.0f} 株')
st.caption(f'現物損益の経路間標準偏差: {stock_std:,.0f} 円 ｜ グリークスはオプション建玉全体の値（現物デルタを除く）')

# Curves at fixed selected day. Conditional hedge paths are separately simulated for each endpoint.
# Graph range is independent of the input slider's full 1–100,000 yen range.
# Center the graph on the selected valuation price (±25%, at least ±500 yen).
chart_half_width=max(float(target)*0.25,500.0)
chart_min=max(1.0,float(target)-chart_half_width)
chart_max=min(100000.0,float(target)+chart_half_width)
xs=np.linspace(chart_min,chart_max,101)
if target not in xs: xs=np.sort(np.append(xs,float(target)))
prices,deltas,gammas,vegas,thetas=bs(xs,K,T,r,q,iv,call)
option_curve=signed_contracts*multiplier*(prices-premium)
stock_curve=[]
with st.spinner('条件付き株価経路とヘッジ損益を計算中...'):
    for v in xs:
        if day==0:
            stock_curve.append(initial_shares*(v-stock_entry))
        else:
            stock_curve.append(simulate(S0,float(v),day,tuple(valuation_dates[:day+1]),expiry,K,r,q,iv,fv,call,signed_contracts,multiplier,initial_shares,stock_entry,min_shares,cost,100,seed)[0])
stock_curve=np.asarray(stock_curve)
series=[('オプション損益',option_curve,option_pl,'円'),('現物ヘッジ損益',stock_curve,stock_pl,'円'),('デルタ',signed_contracts*multiplier*deltas,float(signed_contracts*multiplier*delta),'株相当'),('セータ',signed_contracts*multiplier*thetas,float(signed_contracts*multiplier*theta),'円/日'),('ベガ',signed_contracts*multiplier*vegas,float(signed_contracts*multiplier*vega),'円/IV 1pt'),('ガンマ',signed_contracts*multiplier*gammas,float(signed_contracts*multiplier*gamma),'デルタ株数/円')]
st.subheader('損益・グリークス（横軸：原資産価格）')
for i in range(0,6,2):
    cols=st.columns(2)
    for col,(title,values,point,unit) in zip(cols,series[i:i+2]):
        with col:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=xs,y=values,mode='lines',line=dict(color='#1683ff',width=3),name=title,hovertemplate='株価 %{x:,.0f}円<br>値 %{y:,.3f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[target],y=[point],mode='markers',marker=dict(size=11,color='#ffffff',line=dict(color='#1683ff',width=3)),showlegend=False))
            fig.add_vline(x=target,line_dash='dot',line_color='#aaaaaa')
            fig.update_layout(title=dict(text=f'{title}　{point:+,.3f} {unit}',font=dict(color='#ffffff',size=17)),height=300,margin=dict(l=40,r=15,t=55,b=40),paper_bgcolor='#050608',plot_bgcolor='#050608',font=dict(color='white',size=12),xaxis=dict(title='株価 (円)',showgrid=True,gridcolor='#30343c',tickformat=',.0f'),yaxis=dict(showgrid=True,gridcolor='#30343c',zeroline=True,zerolinecolor='#666'),showlegend=False)
            st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False})

# Replay one of the *same* conditional paths used for the selected endpoint.
# This is an audit trail, not the 100-path mean displayed above.
def replay_hedge_path(path, valuation_dates, expiry_date, K, r, q, iv, call, signed_contracts,
                      multiplier, initial_shares, initial_stock_price, min_shares,
                      cost, lot):
    held=float(initial_shares)
    cash=-held*float(initial_stock_price)
    last=held
    rows=[]
    for d,S in enumerate(path):
        S=float(S)
        remaining=max((expiry_date-valuation_dates[d]).days/365,0)
        _,option_delta,_,_,_=bs(S,K,remaining,r,q,iv,call)
        required=float(-signed_contracts*multiplier*option_delta)
        threshold=max(.25*abs(last),float(min_shares))
        before=held
        trade=0.
        if d>0 and abs(required-last)>=threshold-1e-10:
            desired=float(lot_round(required,lot))
            trade=desired-held
            cash-=trade*S+abs(trade)*cost
            held+=trade
            last=held
        rows.append({'営業日':d,'株価（円）':round(S,2),
                     '必要ヘッジ株数':round(required,2),
                     '調整前株数':int(round(before)),
                     '売買株数（＋買／−売）':int(round(trade)),
                     '調整後株数':int(round(held)),
                     '調整判定': '売買あり' if trade else 'なし',
                     '累積売買コスト（円）':None,
                     '現物評価損益（円）':round(cash+held*S,2)})
    # Reconstruct cumulative transaction costs for auditability.
    running_cost=0.
    for row in rows:
        running_cost+=abs(row['売買株数（＋買／−売）'])*cost
        row['累積売買コスト（円）']=round(running_cost,2)
    return pd.DataFrame(rows)

st.subheader('デルタヘッジ調整履歴')
st.caption('指定した評価日・株価へ到達する100経路のうち、1経路の実際の売買を表示します。平均経路の売買履歴ではありません。')
h1,h2=st.columns([1,2])
with h1:
    path_number=st.number_input('確認する経路番号（1～100）',min_value=1,max_value=100,value=1,step=1)
with h2:
    trades_only=st.checkbox('売買が発生した日だけ表示',value=False)
if day==0:
    selected_path=np.array([S0],dtype=float)
else:
    # Reuse the cached simulation to guarantee matching random paths and P/L.
    _,_,_,all_paths=simulate(S0,float(target),day,tuple(valuation_dates[:day+1]),expiry,K,r,q,iv,fv,call,
        signed_contracts,multiplier,initial_shares,stock_entry,min_shares,cost,100,seed)
    selected_path=all_paths[int(path_number)-1]
history=replay_hedge_path(selected_path,tuple(valuation_dates[:day+1]),expiry,K,r,q,iv,call,signed_contracts,
    multiplier,initial_shares,stock_entry,min_shares,cost,100)
trade_count=int((history['売買株数（＋買／−売）']!=0).sum())
volume=int(history['売買株数（＋買／−売）'].abs().sum())
path_pl=float(history['現物評価損益（円）'].iloc[-1])
hc1,hc2,hc3=st.columns(3)
hc1.metric('この経路の売買回数',f'{trade_count} 回')
hc2.metric('累計売買株数',f'{volume:,} 株')
hc3.metric('この経路の現物損益',f'{path_pl:+,.0f} 円')
shown=history[history['売買株数（＋買／−売）']!=0] if trades_only else history
st.dataframe(shown,hide_index=True,use_container_width=True)
st.download_button('調整履歴をCSVで保存',history.to_csv(index=False).encode('utf-8-sig'),
                   file_name=f'hedge_history_path_{path_number}.csv',mime='text/csv')

with st.expander('評価の前提・注意点（必ず確認）'):
    st.markdown('''- **欧州型ブラック・ショールズ**で価格とグリークスを算出。実際の契約の権利行使方式、乗数、権利落ち・最終売買日は別途確認してください。
- 株価経路は**対数株価のブラウン橋**で、指定日・指定株価に到達する100経路を生成します。日次ショックの予想ボラを使用し、ドリフトは0%です。端点を固定する条件付き分布では、ドリフトの影響は消えます。
- 25%乖離または最低株数の大きい方でヘッジ発動。目標株数を**100株単位**に四捨五入（中間は絶対値で切り上げ）。ヘッジ後もデルタが残る場合があります。
- 現物損益は初期建値からの評価損益に、日次ヘッジ売買キャッシュフローと1株当たり取引コストを反映します。オプション損益は入力建値からの時価評価です。
- 金利・配当利回りはBS評価に使いますが、現物在庫の資金調達費用・配当受払・オプション売買手数料は**現物PLに含みません**。実務のネットPLでは別途必要です。
- 営業日はJPX公式休業日一覧から取得し、土日と休業日を除きます。残存期間とセータは暦日/365で評価します。
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
- 株価グラフの現物PLは、**各横軸株価を終点とする別々の条件付き100経路の平均**です。スライダーを動かしても同一経路の実現PLが変化するという意味ではありません。
- 100経路の推定にはモンテカルロ誤差があります。決算イベント、IVスマイル、板の流動性、スリッページ等は再現していません。''')
