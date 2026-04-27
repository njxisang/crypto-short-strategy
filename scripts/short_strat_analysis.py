#!/usr/bin/env python3
"""
做空组合策略分析：资金费率 × 波动率
聚焦：做空高费率 + 高波动币种，稳定收益
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 配置 ============
DATA_DIR  = "/home/xisang/crypto-new/binance_data"
SWAP_DIR  = f"{DATA_DIR}/binance-swap-candle-csv-1h"
FR_DIR    = f"{DATA_DIR}/binance_funding_rate/usdt"
COIN_DIR  = f"{DATA_DIR}/coin-cap"
TOP_N     = 80
OUTPUT    = "/home/xisang/crypto-new"
COST_RATE = 0.0005  # 0.05% 手续费

def norm(s):
    return s.replace('-','').replace('.csv','').upper()

# ============ 加载数据 ============
print("=" * 70)
print("📂 [1/4] 加载数据...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
coin_files = glob.glob(f"{COIN_DIR}/*.csv")

swap_n2o = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_n2o    = {norm(os.path.basename(f)): os.path.basename(f) for f in fr_files}
coin_n2o  = {norm(os.path.basename(f)): os.path.basename(f) for f in coin_files}

common = set(swap_n2o.keys()) & set(fr_n2o.keys()) & set(coin_n2o.keys())
syms   = sorted(list(common))[:TOP_N]
print(f"  交集: {len(common)}, 分析: {len(syms)}")

all_rows = []
for nsym in syms:
    # K线
    sp = f"{SWAP_DIR}/{swap_n2o[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={
        'quote_volume': 'qv', 'volume': 'vol',
        'taker_buy_base_asset_volume': 'taker_buy_vol'
    })
    df_s['date'] = df_s['candle_begin_time'].dt.date
    dag = df_s.groupby('date').agg(
        close=('close','last'),
        high=('high','max'),
        low=('low','min'),
        qv=('qv','sum'),
        vol=('vol','sum'),
        taker_buy_vol=('taker_buy_vol','sum'),
    ).reset_index()

    # 资金费率
    fp = f"{FR_DIR}/{fr_n2o[nsym]}"
    df_fr = pd.read_csv(fp, parse_dates=['time'])
    df_fr['date'] = df_fr['time'].dt.date
    dag_fr = df_fr.groupby('date')['fundingRate'].mean().reset_index()
    dag_fr.columns = ['date','fr']

    # 链上
    cp = f"{COIN_DIR}/{coin_n2o[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    df_c['date'] = df_c['candle_begin_time'].dt.date
    dag_c = df_c.groupby('date')['circulating_supply'].last().reset_index()
    dag_c.columns = ['date','supply']

    dag['date'] = pd.to_datetime(dag['date'])
    dag_fr['date'] = pd.to_datetime(dag_fr['date'])
    dag_c['date'] = pd.to_datetime(dag_c['date'])

    m = dag.merge(dag_fr, on='date', how='left')
    m = m.merge(dag_c, on='date', how='left')
    m['symbol_norm'] = nsym
    all_rows.append(m)

df = pd.concat(all_rows, ignore_index=True)
df = df.sort_values(['symbol_norm','date']).reset_index(drop=True)

# 基础指标
df['ret']   = df.groupby('symbol_norm')['close'].pct_change()
df['fret']  = df.groupby('symbol_norm')['ret'].shift(-1)
df['fr']    = df['fr'].fillna(0)
df['supply'] = df['supply'].fillna(method='ffill').fillna(method='bfill')

# 因子
df['vol_7d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())
df['vol_14d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(14, min_periods=7).std())
df['vol_30d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(30, min_periods=15).std())
df['mom_7d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())
df['mom_14d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(14, min_periods=7).mean())
df['mom_30d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(30, min_periods=15).mean())
df['fr_3d']   = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['fr_7d']   = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(7, min_periods=3).mean())
df['fr_14d']  = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(14, min_periods=7).mean())

# 截面标准化
def cz(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

df['vol_z']  = df.groupby('date')['vol_7d'].transform(cz, df['vol_7d'])
df['vol14_z'] = df.groupby('date')['vol_14d'].transform(cz, df['vol_14d'])
df['mom_z']  = df.groupby('date')['mom_7d'].transform(cz, df['mom_7d'])
df['mom14_z'] = df.groupby('date')['mom_14d'].transform(cz, df['mom_14d'])
df['fr_z']   = df.groupby('date')['fr'].transform(cz, df['fr'])
df['fr7_z']  = df.groupby('date')['fr_7d'].transform(cz, df['fr_7d'])
df['fr14_z'] = df.groupby('date')['fr_14d'].transform(cz, df['fr_14d'])

# 复合因子
df['fr_x_vol']    = df['fr_z'] * df['vol_z']         # 费率×波动率
df['fr7_x_vol']  = df['fr7_z'] * df['vol_z']         # 7日费率×波动率
df['fr14_x_vol'] = df['fr14_z'] * df['vol_z']        # 14日费率×波动率
df['fr7_x_vol14']= df['fr7_z'] * df['vol14_z']       # 7日费率×14日波动率
df['mom_x_vol']  = df['mom_z'] * df['vol_z']         # 动量×波动率
df['fr7_div_vol']= df['fr7_z'] / (df['vol_z'] + 1e-9)  # 费率/波动率 (高费率低波动 = 做空机会)

# 截面分档标记
df['vol_q']  = df.groupby('date')['vol_7d'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
df['fr_q']   = df.groupby('date')['fr'].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['close','ret','fret'])
df = df[df['date'] >= '2021-01-01']

print(f"  {len(df):,} 条 | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ 核心分析 ============
print("\n" + "=" * 70)
print("🎯 核心分析：做空高费率 + 高波动 组合")
print("=" * 70)

# ---- A. 按 FR×Vol 复合因子分档 ----
print("\n📊 [A] FR×Vol 复合因子分档 IC (预测次日收益)")
print("-" * 70)

combo_factors = ['fr_x_vol', 'fr7_x_vol', 'fr14_x_vol', 'fr7_x_vol14', 'fr7_div_vol']
factor_cols_all = combo_factors + ['fr_z', 'fr7_z', 'vol_z', 'vol14_z', 'mom_z', 'mom14_z']

ic_rows = []
for fc in factor_cols_all:
    if fc not in df.columns:
        continue
    for dt, grp in df.groupby('date'):
        sub = grp.dropna(subset=[fc,'fret'])
        if len(sub) < 15:
            continue
        try:
            ic, _ = spearmanr(sub[fc], sub['fret'], nan_policy='omit')
        except:
            ic = 0
        if np.isnan(ic):
            ic = 0
        ic_rows.append({'date': dt, 'factor': fc, 'ic': ic, 'vol': sub['vol_7d'].mean(), 'fr': sub['fr'].mean()})

ic_df = pd.DataFrame(ic_rows)

# IC 统计
ic_stats = []
for fc in ic_df['factor'].unique():
    sub = ic_df[ic_df['factor'] == fc]['ic'].dropna()
    if len(sub) < 30:
        continue
    ic_stats.append({
        'factor': fc,
        'IC_mean': round(sub.mean(), 5),
        'IC_std':  round(sub.std(), 5),
        'IC_IR':   round(sub.mean()/(sub.std()+1e-9), 4),
        'IC_pos%': round((sub > 0).mean()*100, 1),
        'N': len(sub)
    })

ic_stats_df = pd.DataFrame(ic_stats).sort_values('IC_IR', key=abs, ascending=False)
print(ic_stats_df.to_string(index=False))

# ---- B. 2×2 分组分析：FR × Vol ----
print("\n📊 [B] 2×2 分组：FR(高/低) × Vol(高/低) — 做空组每日收益")
print("-" * 70)

# 先按 median 分割
df['vol_median'] = df.groupby('date')['vol_7d'].transform('median')
df['fr_median']  = df.groupby('date')['fr'].transform('median')
df['vol Regime'] = np.where(df['vol_7d'] > df['vol_median'], 'HighVol', 'LowVol')
df['fr Regime']  = np.where(df['fr'] > df['fr_median'], 'HighFR', 'LowFR')

# 分组未来收益
results = df.groupby(['date','vol Regime','fr Regime'])['fret'].mean().reset_index()
results['combo'] = results['vol Regime'] + ' × ' + results['fr Regime']

# 透视
pivot = results.pivot_table(index='combo', columns='date', values='fret').T
pivot['combo'] = pivot.index

# 做空高FR组每日收益
combo_ret = {}
for combo in ['HighVol × HighFR', 'HighVol × LowFR', 'LowVol × HighFR', 'LowVol × LowFR']:
    r = results[results['combo'] == combo].set_index('date')['fret']
    combo_ret[combo] = r

# 计算各组统计
for combo, s in combo_ret.items():
    n = len(s)
    cum = np.exp(np.log1p(s).sum()) - 1
    ann = (1+cum)**(365.25/n) - 1
    sharpe = s.mean()/s.std()*np.sqrt(365) if s.std() > 0 else 0
    print(f"  {combo:20s}: 日均={s.mean()*100:+.3f}%  年化={ann*100:+.1f}%  Sharpe={sharpe:.2f}  WinRate={(s>0).mean()*100:.1f}%")

# ---- C. 做空最优组合：Q5(高FR+高Vol) vs Q1 ----
print("\n📊 [C] Q5(高费率+高波动) 做空 vs Q1 收益对比")
print("-" * 70)

# 找高FR且高Vol的币：每日做空top20% fr且top20% vol的币
top_pct = 0.2
n_top   = max(3, int(TOP_N * top_pct))

short_daily = {  # 做空信号
    'highFR_only':   {'ret': [], 'fr_inc': [], 'date': []},
    'highVol_only':  {'ret': [], 'fr_inc': [], 'date': []},
    'highFR+Vol':    {'ret': [], 'fr_inc': [], 'date': []},
    'highFR_volAdj': {'ret': [], 'fr_inc': [], 'date': []},
}

for dt, grp in df.groupby('date'):
    if len(grp) < 20:
        continue
    sub = grp.dropna(subset=['fret','fr','vol_7d'])
    if len(sub) < 20:
        continue

    sub_s = sub.sort_values('fr')
    high_fr   = sub_s.tail(n_top)
    low_fr    = sub_s.head(n_top)

    sub_v = sub.sort_values('vol_7d')
    high_vol  = sub_v.tail(n_top)
    low_vol   = sub_v.head(n_top)

    # 高FR∩高Vol
    both = sub[(sub['fr'] >= sub['fr'].quantile(1-top_pct)) & (sub['vol_7d'] >= sub['vol_7d'].quantile(1-top_pct))]
    both = both.sort_values('fr')
    high_fr_vol = both.head(n_top) if len(both) >= n_top else both

    # 费率调整后做空收益
    fr_inc = sub['fr'].clip(upper=0).abs()  # 做空时获得的费率补贴
    cost   = COST_RATE

    # highFR_only 做空
    if len(high_fr) >= n_top:
        r  = (-high_fr['fret'] + fr_inc.loc[high_fr.index] - cost).mean()
        short_daily['highFR_only']['ret'].append(r)
        short_daily['highFR_only']['fr_inc'].append(fr_inc.loc[high_fr.index].mean())
        short_daily['highFR_only']['date'].append(dt)

    # highVol_only 做空
    if len(high_vol) >= n_top:
        r  = (-high_vol['fret'] + fr_inc.loc[high_vol.index] - cost).mean()
        short_daily['highVol_only']['ret'].append(r)
        short_daily['highVol_only']['fr_inc'].append(fr_inc.loc[high_vol.index].mean())
        short_daily['highVol_only']['date'].append(dt)

    # highFR+Vol 做空 (交集)
    if len(high_fr_vol) >= 2:
        r  = (-high_fr_vol['fret'] + fr_inc.loc[high_fr_vol.index] - cost).mean()
        short_daily['highFR+Vol']['ret'].append(r)
        short_daily['highFR+Vol']['fr_inc'].append(fr_inc.loc[high_fr_vol.index].mean())
        short_daily['highFR+Vol']['date'].append(dt)

    # highFR_volAdj: fr/vol 比值最高的 (性价比)
    sub_copy = sub.copy()
    sub_copy['fr_vol_ratio'] = sub_copy['fr'] / (sub_copy['vol_7d'] + 1e-9)
    sub_copy = sub_copy.sort_values('fr_vol_ratio', ascending=False)
    high_fr_vol_adj = sub_copy.head(n_top)
    if len(high_fr_vol_adj) >= 2:
        r  = (-high_fr_vol_adj['fret'] + fr_inc.loc[high_fr_vol_adj.index] - cost).mean()
        short_daily['highFR_volAdj']['ret'].append(r)
        short_daily['highFR_volAdj']['fr_inc'].append(fr_inc.loc[high_fr_vol_adj.index].mean())
        short_daily['highFR_volAdj']['date'].append(dt)

print(f"  {'策略':20s} {'日均收益':>10s} {'费率补贴':>10s} {'年化':>8s} {'Sharpe':>8s} {'胜率':>6s} {'天数':>6s}")
print("  " + "-" * 78)
for name, d in short_daily.items():
    if not d['date']:
        continue
    rets   = np.array(d['ret'])
    fr_inc = np.array(d['fr_inc'])
    n = len(rets)
    cum = np.exp(np.log1p(rets).sum()) - 1
    ann = (1+cum)**(365.25/n) - 1
    sh  = rets.mean()/rets.std()*np.sqrt(365) if rets.std() > 0 else 0
    wr  = (rets > 0).mean()*100
    avg_fr = fr_inc.mean()*100
    print(f"  {name:20s} {rets.mean()*100:>+10.4f}% {avg_fr:>+10.4f}% {ann*100:>+8.1f}% {sh:>8.2f} {wr:>6.1f}% {n:>6d}")

# ---- D. 分年度看做空策略稳定性 ----
print("\n📊 [D] 分年度做空策略年化收益")
print("-" * 70)

yearly_short = {}
for name, d in short_daily.items():
    if not d['date']:
        continue
    df_d = pd.DataFrame(d)
    df_d['year'] = pd.to_datetime(df_d['date']).dt.year
    yr = df_d.groupby('year')['ret'].apply(
        lambda x: (np.exp(np.log1p(x).sum())**(365.25/len(x)) - 1) * 100
    )
    yearly_short[name] = yr

if yearly_short:
    yr_df = pd.DataFrame(yearly_short)
    yr_df['avg'] = yr_df.mean(axis=1)
    print(yr_df.round(2).to_string())

# ---- E. 月度做空收益率热力图 ----
print("\n📊 [E] 做空高FR+高Vol 组合月度收益率 (%)")
print("-" * 70)

d = short_daily.get('highFR+Vol', {})
if d['date']:
    df_d = pd.DataFrame(d)
    df_d['date'] = pd.to_datetime(df_d['date'])
    df_d['ym'] = df_d['date'].dt.to_period('M')
    monthly = df_d.groupby('ym')['ret'].apply(lambda x: x.mean()*100)
    monthly.name = 'highFR+Vol'
    monthly.index = monthly.index.astype(str)
    # 显示最近24个月
    print(monthly.tail(24).round(3).to_string())

# ---- F. 择时：只在高费率区间做空 ----
print("\n📊 [F] 择时效果：只在全市场平均费率>阈值时做空")
print("-" * 70)

# 计算每日全市场平均费率
daily_avg_fr = df.groupby('date')['fr'].mean()

for threshold in [0, 0.0001, 0.0002, 0.0003, 0.0005]:
    threshold_label = f"FR>{threshold:.4f}"
    mask = daily_avg_fr[daily_avg_fr > threshold].index
    # 做空高FR+Vol
    d = short_daily.get('highFR+Vol', {})
    if d['date']:
        df_d = pd.DataFrame(d)
        df_d = df_d[df_d['date'].isin(mask)]
        if len(df_d) > 30:
            rets = df_d['ret'].values
            n = len(rets)
            cum = np.exp(np.log1p(rets).sum()) - 1
            ann = (1+cum)**(365.25/n) - 1
            sh  = rets.mean()/rets.std()*np.sqrt(365) if rets.std() > 0 else 0
            wr  = (rets > 0).mean()*100
            print(f"  {threshold_label:15s}: 年化={ann*100:>+7.1f}%  Sharpe={sh:>6.2f}  胜率={wr:>5.1f}%  n_days={n}")

# ---- G. 最优组合：fr/vol比值 × fr趋势 ----
print("\n📊 [G] 扩展因子 IC — 找最优做空组合")
print("-" * 70)

# 额外复合因子
df['fr_spread']   = df['fr_7d'] - df['fr_14d']  # 费率扩张
df['fr_vol_adj']  = df['fr_7d'] / (df['vol_7d'] + 1e-9)  # 费率/波动率
df['neg_mom']     = -df['mom_7d']  # 负动量 = 做空信号
df['neg_mom14']   = -df['mom_14d']
df['neg_mom_vol'] = -df['mom_7d'] * df['vol_7d']  # 负动量×波动率

def cz(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

for fc_raw in ['fr_spread', 'fr_vol_adj', 'neg_mom', 'neg_mom14', 'neg_mom_vol']:
    if fc_raw in df.columns:
        df[fc_raw + '_z'] = df.groupby('date')[fc_raw].transform(cz, df[fc_raw])

extended_factors = [c for c in ['fr_spread_z','fr_vol_adj_z','neg_mom_z','neg_mom14_z','neg_mom_vol_z'] if c in df.columns]
extended_factors += [c for c in ['fr7_x_vol','fr14_x_vol','fr7_div_vol'] if c in df.columns]

ic_rows2 = []
for fc in extended_factors:
    for dt, grp in df.groupby('date'):
        sub = grp.dropna(subset=[fc,'fret'])
        if len(sub) < 15:
            continue
        try:
            ic, _ = spearmanr(sub[fc], sub['fret'], nan_policy='omit')
        except:
            ic = 0
        if np.isnan(ic):
            ic = 0
        ic_rows2.append({'date': dt, 'factor': fc, 'ic': ic})

ic_df2 = pd.DataFrame(ic_rows2)

ext_stats = []
for fc in ic_df2['factor'].unique():
    sub = ic_df2[ic_df2['factor'] == fc]['ic'].dropna()
    if len(sub) < 30:
        continue
    ext_stats.append({
        'factor': fc,
        'IC_mean': round(sub.mean(), 5),
        'IC_std':  round(sub.std(), 5),
        'IC_IR':   round(sub.mean()/(sub.std()+1e-9), 4),
        'IC_pos%': round((sub > 0).mean()*100, 1),
        'N': len(sub)
    })

ext_stats_df = pd.DataFrame(ext_stats).sort_values('IC_IR', key=abs, ascending=False)
print(ext_stats_df.to_string(index=False))

# 分年度
if not ext_stats_df.empty:
    ic_df2['year'] = pd.to_datetime(ic_df2['date']).dt.year
    yr2 = ic_df2.groupby(['year','factor']).agg(IC_mean=('ic','mean')).reset_index()
    pivot2 = yr2.pivot_table(index='factor', columns='year', values='IC_mean').dropna(thresh=5)
    print("\n分年度 IC 均值:")
    print(pivot2.round(4).to_string())

# ---- H. 最优做空组合策略模拟 ----
print("\n" + "=" * 70)
print("📋 [H] 最优做空组合策略回测 (2021-2026, 每周rebalance)")
print("=" * 70)

# 使用 fr_vol_adj_z 因子 (费率/波动率比值) 做空
best_factor = 'fr_vol_adj_z' if 'fr_vol_adj_z' in df.columns else 'fr7_div_vol'
print(f"  使用因子: {best_factor}")

# 周频 rebalance
df['week'] = df['date'].dt.isocalendar().week.astype(str) + '-' + df['date'].dt.year.astype(str)
weekly_rets = []
weekly_dates = sorted(df['week'].unique())

for i, wk in enumerate(weekly_dates):
    end_wk = df[df['week'] == wk]['date'].max()
    next_wk = weekly_dates[i+1] if i+1 < len(weekly_dates) else None
    if next_wk is None:
        continue
    next_dt = df[df['week'] == next_wk]['date'].min()

    # 当天截面
    day_data = df[df['date'] == end_wk].dropna(subset=[best_factor,'fret'])
    if len(day_data) < 20:
        continue

    # 做空因子值最高(费率/波动率最高)的币
    day_data_s = day_data.sort_values(best_factor, ascending=False)
    n_short = max(3, int(len(day_data_s) * 0.2))
    short_coins = day_data_s.head(n_short)

    # 计算做空收益(持有到下周)
    wk_data = df[(df['date'] >= end_wk) & (df['date'] < next_dt)]
    for _, coin in short_coins.iterrows():
        sym = coin['symbol_norm']
        coin_wk = wk_data[wk_data['symbol_norm'] == sym]
        if len(coin_wk) > 0:
            ret = coin_wk['ret'].mean()
            fr_inc = coin_wk['fr'].clip(upper=0).abs().mean()
            weekly_rets.append({
                'date': end_wk,
                'ret': -ret + fr_inc * 7/8 - COST_RATE,
                'sym': sym
            })

wr_df = pd.DataFrame(weekly_rets)
if not wr_df.empty:
    # 组合收益
    port = wr_df.groupby('date')['ret'].mean()
    n = len(port)
    cum = np.exp(np.log1p(port).sum()) - 1
    ann = (1+cum)**(52/n) - 1
    sh = port.mean()/port.std()*np.sqrt(52) if port.std() > 0 else 0
    wr = (port > 0).mean()*100
    print(f"  年化收益: {ann*100:+.1f}%")
    print(f"  Sharpe:   {sh:.2f}")
    print(f"  胜率:     {wr:.1f}%")
    print(f"  交易周数: {n}")
    # 分年
    port_df = port.reset_index()
    port_df['date'] = pd.to_datetime(port_df['date'])
    port_df['year'] = port_df['date'].dt.year
    yearly_p = port_df.groupby('year').apply(lambda x: (np.exp(np.log1p(x['ret']).sum())**(365.25/len(x))-1)*100)
    print(f"  分年收益: {dict(yearly_p.round(1))}")

print("\n✅ 分析完成")
