#!/usr/bin/env python3
"""
加密货币多因子 IC/IR 完整分析 — 2021-2026
聚焦: 长线做空策略因子筛选
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 配置 ============
DATA_DIR   = "/home/xisang/crypto-new/binance_data"
SWAP_DIR   = f"{DATA_DIR}/binance-swap-candle-csv-1h"
FR_DIR     = f"{DATA_DIR}/binance_funding_rate/usdt"
COIN_DIR   = f"{DATA_DIR}/coin-cap"
TOP_N      = 80        # 截面TOP币种
OUTPUT     = "/home/xisang/crypto-new"
REBAL_H    = 24        # 预测周期 (小时) — 长线可用 72/168
HALFLIFE   = 30        # IC衰减半衰期 (天)
COST_RATE  = 0.0005    # 交易费率
TOP_PCT    = 0.2       # 多空各20%币种
# 预测收益的N小时后的收益
FUTURE_H   = REBAL_H

def norm(s):
    """标准化币种名称"""
    return s.replace('-','').replace('.csv','').replace('.CSV','').upper()

# ============ Step 1: 加载数据 ============
print("=" * 70)
print(f"📂 [1/5] 加载数据 (重仓周期={REBAL_H}h)")
print("=" * 70)

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
coin_files = glob.glob(f"{COIN_DIR}/*.csv")

swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_norm2orig   = {norm(os.path.basename(f).replace('USDT','USDT')): os.path.basename(f) for f in fr_files}
coin_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in coin_files}

# 交集: 三数据源都有
common = set(swap_norm2orig.keys()) & set(fr_norm2orig.keys()) & set(coin_norm2orig.keys())
target_syms = sorted(list(common))[:TOP_N]
print(f"  三数据源交集: {len(common)}, 分析: {len(target_syms)}")

# 按时间聚合: 日频 (用日期分组)
all_rows = []
for nsym in target_syms:
    # === K线数据 ===
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
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

    # === 资金费率 ===
    fp = f"{FR_DIR}/{fr_norm2orig[nsym]}"
    df_fr = pd.read_csv(fp, parse_dates=['time'])
    df_fr['date'] = df_fr['time'].dt.date
    dag_fr = df_fr.groupby('date')['fundingRate'].mean().reset_index()
    dag_fr.columns = ['date','fr']

    # === 链上数据 ===
    cp = f"{COIN_DIR}/{coin_norm2orig[nsym]}"
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

# 未来收益
df['ret']   = df.groupby('symbol_norm')['close'].pct_change()
df['fret']  = df.groupby('symbol_norm')['ret'].shift(-1)  # 1日后收益
# N小时后收益 (日内重仓用)
df['fret_h'] = df.groupby('symbol_norm')['close'].pct_change(FUTURE_H // 24 if FUTURE_H >= 24 else 1).shift(-FUTURE_H // 24 if FUTURE_H >= 24 else -1)
df['fret_h'] = df.groupby('symbol_norm')['fret_h'].shift(-(FUTURE_H % 24))

df['fr']     = df['fr'].fillna(0)
df['supply'] = df['supply'].fillna(method='ffill').fillna(method='bfill')

print(f"  {len(df):,} 条 | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 2: 构建因子 ============
print("\n🔧 [2/5] 因子构建...")

def cz(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

# --- 动量类 ---
df['ret_7d']   = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).sum())
df['ret_14d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(14, min_periods=7).sum())
df['ret_30d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(30, min_periods=15).sum())
df['mom_7d']   = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())
df['mom_14d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(14, min_periods=7).mean())

# --- 波动率类 ---
df['vol_7d']   = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())
df['vol_14d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(14, min_periods=7).std())
df['vol_30d']  = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(30, min_periods=15).std())
# 高波动区分
df['vol_high'] = (df['vol_7d'] > df.groupby('date')['vol_7d'].transform('median')).astype(int)

# --- 资金费率类 ---
df['fr_3d']    = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['fr_7d']    = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(7, min_periods=3).mean())
df['fr_14d']   = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(14, min_periods=7).mean())
df['fr_sign']  = np.sign(df['fr'])
df['fr_high']  = (df['fr'] > 0).astype(int)  # 正费率 = 多头支付
# 费率趋势: 3日均线相对7日均线
df['fr_trend'] = df['fr_3d'] - df['fr_7d']
# 8h 节律: 费率在8h周期的位置
df['fr_cycle'] = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean() - x.rolling(8, min_periods=4).mean())

# --- 订单流 ---
df['net_flow'] = df['taker_buy_vol'] / (df['vol'] + 1e-9)
df['flow_3d']  = df.groupby('symbol_norm')['net_flow'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['flow_7d']  = df.groupby('symbol_norm')['net_flow'].transform(lambda x: x.rolling(7, min_periods=3).mean())

# --- 链上 ---
df['supply_chg'] = df.groupby('symbol_norm')['supply'].pct_change(7)
df['supply_chg14'] = df.groupby('symbol_norm')['supply'].pct_change(14)
df['log_supply'] = np.log1p(df['supply'])

# --- 偏度/峰度 (高阶矩) ---
def rolling_skew(x, w):
    return x.rolling(w, min_periods=w//2).apply(lambda a: pd.Series(a).skew(), raw=True)
def rolling_kurt(x, w):
    return x.rolling(w, min_periods=w//2).apply(lambda a: pd.Series(a).kurt(), raw=True)

df['skew_14d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: rolling_skew(x, 14))
df['skew_30d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: rolling_skew(x, 30))
df['kurt_14d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: rolling_kurt(x, 14))
df['kurt_30d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: rolling_kurt(x, 30))

# --- 截面标准化因子 ---
factors_raw = {
    'fr_z':        'fr',
    'fr_trend_z':  'fr_trend',
    'fr_sign':     'fr',
    'vol_7d_z':    'vol_7d',
    'vol_14d_z':   'vol_14d',
    'mom_7d_z':    'mom_7d',
    'mom_14d_z':   'mom_14d',
    'mom_30d_z':   'mom_30d',
    'supply_z':    'log_supply',
    'supply_chg_z':'supply_chg',
    'flow_z':      'net_flow',
    'fr_cycle_z':  'fr_cycle',
    'skew_z':      'skew_14d',
    'kurt_z':      'kurt_14d',
}
for fc, src in factors_raw.items():
    if src in df.columns:
        df[fc] = df.groupby('date')[src].transform(cz, df[src])

factor_cols = list(factors_raw.keys())
# 去掉不存在的列
factor_cols = [c for c in factor_cols if c in df.columns]
print(f"  共 {len(factor_cols)} 个因子: {factor_cols}")

# ============ Step 3: 清理 & 过滤 ============
print("\n🧹 [3/5] 数据清理...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['close','ret','fret'])

# 只保留 2021-01-01 以后
df = df[df['date'] >= '2021-01-01']
print(f"  2021后: {len(df):,} 条 | {df['symbol_norm'].nunique()} 币")

# ============ Step 4: IC/IR 计算 ============
print("\n📊 [4/5] IC/IR 计算...")
from scipy.stats import spearmanr

ic_rows = []
strat_daily = {f: {'long':[], 'short':[], 'ls':[], 'date':[]} for f in factor_cols}
strat_daily['fr_q'] = {'Q1_long':[], 'Q5_short':[], 'Q1Q5_ls':[], 'date':[]}

top_n_coins = max(5, int(TOP_PCT * df['symbol_norm'].nunique()))
bot_n_coins = top_n_coins

# 权重
df['weight'] = 1.0 / np.log1p(df['supply'])

for dt, grp in df.groupby('date'):
    if len(grp) < 10:
        continue
    sub = grp.dropna(subset=['fret'] + factor_cols)
    if len(sub) < 10:
        continue

    # --- IC ---
    for fc in factor_cols:
        try:
            ic, _ = spearmanr(sub[fc], sub['fret'], nan_policy='omit')
        except:
            ic = 0
        wt = sub.loc[sub[fc].notna(), 'weight'].mean() if sub[fc].notna().any() else 1.0
        ic_rows.append({'date': dt, 'factor': fc, 'ic': ic if not np.isnan(ic) else 0, 'weight': wt})

    # --- 多空组合 (因子方向: IC<0 则做空高因子值) ---
    for fc in factor_cols:
        sub_s = sub.dropna(subset=[fc,'fret']).sort_values(fc)
        if len(sub_s) < top_n_coins * 2:
            continue
        long_coin  = sub_s.tail(top_n_coins)
        short_coin = sub_s.head(bot_n_coins)

        # 做空费率补贴 (做空高费率币种额外收益)
        fr_inc_short = sub_s['fr'].clip(upper=0).abs() * (REBAL_H / 8)  # 费率随时间累积
        cost = COST_RATE * (REBAL_H / 24)

        long_ret  = long_coin['fret'].mean()
        short_ret = (-short_coin['fret'] + fr_inc_short.loc[short_coin.index] - cost).mean()

        strat_daily[fc]['long'].append(long_ret)
        strat_daily[fc]['short'].append(short_ret)
        strat_daily[fc]['ls'].append(0.5 * long_ret + 0.5 * short_ret)
        strat_daily[fc]['date'].append(dt)

    # --- fr_sign 分档 ---
    sub_q = sub.dropna(subset=['fr_sign','fret']).sort_values('fr')
    if len(sub_q) >= top_n_coins * 2:
        q1 = sub_q.head(top_n_coins)   # 最低费率 (做空信号)
        q5 = sub_q.tail(top_n_coins)   # 最高费率 (做多信号)
        fr_inc_short = sub_q['fr'].clip(upper=0).abs() * (REBAL_H / 8) - cost
        q1_long = (-q1['fret'] + fr_inc_short.loc[q1.index]).mean()
        q5_short = (-q5['fret'] + fr_inc_short.loc[q5.index]).mean()
        strat_daily['fr_q']['Q1_long'].append(q1_long)
        strat_daily['fr_q']['Q5_short'].append(q5_short)
        strat_daily['fr_q']['Q1Q5_ls'].append(0.5 * q1_long + 0.5 * q5_short)
        strat_daily['fr_q']['date'].append(dt)

ic_df = pd.DataFrame(ic_rows)

# ============ IC 统计 ============
print("\n" + "=" * 70)
print("📋 IC/IR 总体统计 (2021-2026)")
print("=" * 70)

ic_stats = []
for fc in factor_cols:
    sub_ic = ic_df[ic_df['factor'] == fc]['ic'].dropna()
    if len(sub_ic) < 30:
        continue
    ic_mean   = sub_ic.mean()
    ic_std    = sub_ic.std()
    ic_ir     = ic_mean / (ic_std + 1e-9)
    ic_pos    = (sub_ic > 0).mean()
    ic_abs    = sub_ic.abs().mean()

    ic_stats.append({
        'factor': fc,
        'IC_mean': round(ic_mean, 5),
        'IC_std':  round(ic_std, 5),
        'IC_IR':   round(ic_ir, 4),
        'IC_pos%': round(ic_pos*100, 1),
        'IC_abs':  round(ic_abs, 5),
        'N': len(sub_ic)
    })

ic_stats_df = pd.DataFrame(ic_stats).sort_values('IC_IR', key=abs, ascending=False)
print(ic_stats_df.to_string(index=False))

# ============ 多空收益 ============
print("\n" + "=" * 70)
print(f"📋 多空策略收益 (Rebal={REBAL_H}h, 2021-2026)")
print("=" * 70)

ls_results = []

# 常规因子
for fc in factor_cols:
    dates = strat_daily[fc]['date']
    if not dates:
        continue
    longs  = np.array(strat_daily[fc]['long'])
    shorts = np.array(strat_daily[fc]['short'])
    ls     = np.array(strat_daily[fc]['ls'])
    n_days = len(ls)
    ls_cum = np.exp(np.log1p(ls).sum()) - 1
    annual = (1 + ls_cum) ** (365.25 / n_days) - 1 if n_days > 0 else 0
    ls_results.append({
        'factor': fc,
        'AnnRet%': round(annual * 100, 2),
        'Long%':   round((np.exp(np.log1p(longs).sum())**(365.25/len(longs)) - 1) * 100, 2),
        'Short%':  round((np.exp(np.log1p(shorts).sum())**(365.25/len(shorts)) - 1) * 100, 2),
        'Sharpe':  round(ls.mean()/ls.std()*np.sqrt(365.25/REBAL_H*24), 3) if ls.std() > 0 else 0,
        'WinRate': round((ls > 0).mean()*100, 1),
        'N_days':  n_days
    })

# fr_q 单独处理
fd = strat_daily['fr_q']
if fd['date']:
    q1l   = np.array(fd['Q1_long'])
    q5s   = np.array(fd['Q5_short'])
    ls    = np.array(fd['Q1Q5_ls'])
    n_days = len(ls)
    annual = (np.exp(np.log1p(ls).sum())**(365.25/n_days) - 1) * 100 if n_days > 0 else 0
    ls_results.append({
        'factor': 'fr_q',
        'AnnRet%': round(annual, 2),
        'Long%':   round((np.exp(np.log1p(q1l).sum())**(365.25/len(q1l)) - 1) * 100, 2),
        'Short%':  round((np.exp(np.log1p(q5s).sum())**(365.25/len(q5s)) - 1) * 100, 2),
        'Sharpe':  round(ls.mean()/ls.std()*np.sqrt(365.25/REBAL_H*24), 3) if ls.std() > 0 else 0,
        'WinRate': round((ls > 0).mean()*100, 1),
        'N_days':  n_days
    })

ls_df = pd.DataFrame(ls_results).sort_values('AnnRet%', ascending=False)
print(ls_df.to_string(index=False))

# ============ 分年度 IC ============
print("\n" + "=" * 70)
print("📋 分年度 IC 统计")
print("=" * 70)

ic_df['year'] = ic_df['date'].dt.year
yearly = ic_df.groupby(['year','factor']).agg(
    IC_mean=('ic','mean'),
    IC_std=('ic','std'),
    IC_IR=('ic', lambda x: x.mean()/(x.std()+1e-9)),
    N=('ic','count')
).reset_index()

# 透视表
pivot = yearly.pivot_table(index='factor', columns='year', values='IC_IR', aggfunc='first')
# 只保留有完整年份的因子
pivot = pivot.dropna(thresh=len(pivot.columns)-1)
pivot['avg_IR'] = pivot.mean(axis=1)
pivot = pivot.sort_values('avg_IR', key=abs, ascending=False)
print(pivot.round(3).to_string())

# ============ 月度 IC 热力图数据 ============
print("\n" + "=" * 70)
print("📋 月度 IC 均值 (部分月份)")
print("=" * 70)

ic_df['ym'] = ic_df['date'].dt.to_period('M')
monthly_ic = ic_df.groupby(['ym','factor'])['ic'].mean().unstack('factor')
# 只显示关键因子
key_factors = ['fr_z','fr_trend_z','vol_7d_z','mom_7d_z','mom_14d_z','mom_30d_z','supply_z','flow_z','skew_z']
key_factors = [f for f in key_factors if f in monthly_ic.columns]
print(monthly_ic[key_factors].tail(24).round(4).to_string())

# ============ 保存结果 ============
ic_stats_df.to_csv(f"{OUTPUT}/full_ic_stats_{REBAL_H}h.csv", index=False)
ls_df.to_csv(f"{OUTPUT}/full_ls_returns_{REBAL_H}h.csv", index=False)
pivot.round(4).to_csv(f"{OUTPUT}/full_yearly_ic_ir_{REBAL_H}h.csv")
monthly_ic[key_factors].to_csv(f"{OUTPUT}/full_monthly_ic_{REBAL_H}h.csv")

print(f"\n✅ 结果保存: full_ic_stats_{REBAL_H}h.csv, full_ls_returns_{REBAL_H}h.csv")

# ============ 做空专项分析 ============
print("\n" + "=" * 70)
print("🎯 做空专项 — 负IC因子 (这些因子做空高值端赚钱)")
print("=" * 70)

neg_ic = ic_stats_df[ic_stats_df['IC_mean'] < 0].sort_values('IC_mean')
print(neg_ic.to_string(index=False))

# 分年度看负IC稳定性
neg_factors = neg_ic['factor'].tolist()
pivot_neg = yearly.pivot_table(index='factor', columns='year', values='IC_mean', aggfunc='first')
pivot_neg = pivot_neg.loc[pivot_neg.index.isin(neg_factors)]
print("\n分年度负IC均值:")
print(pivot_neg.round(4).to_string())
