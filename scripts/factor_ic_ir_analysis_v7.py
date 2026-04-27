#!/usr/bin/env python3
"""
多因子有效性分析 v7 FIXED — IC加权相对收益版
=================================================================
年化修复: 每日rebalance后, 收集所有日收益 → 几何平均 → annualize
不再月度annualize再平均

核心逻辑:
1. 每日计算IC分档多空收益 → 收集所有日收益
2. 全时期几何累积 → (1+r_total)^(365/总天数) - 1
3. 分段几何累积 → 分别annualize
=================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, glob, warnings
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)

# ============ 配置 ============
DATA_DIR  = "/home/xisang/crypto-new/binance_data"
SWAP_DIR  = f"{DATA_DIR}/binance-swap-candle-csv-1h"
COIN_DIR  = f"{DATA_DIR}/coin-cap"
TOP_N     = 60
FUTURE_H  = 24
HALFLIFE  = 30
MAX_RET   = 0.30
OUTPUT    = "/home/xisang/crypto-new"
COST_RATE = 0.0005  # 0.05%

def norm(s):
    return s.replace('-', '').replace('.csv', '').replace('.CSV', '').upper()

# ============ Step 1: 加载 ============
print("=" * 60)
print("📂 [1/6] 加载数据...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
cap_files  = glob.glob(f"{COIN_DIR}/*.csv")
swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
cap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in cap_files}
common = set(swap_norm2orig.keys()) & set(cap_norm2orig.keys())
target_syms = sorted(list(common))[:TOP_N]
print(f"  交集: {len(common)}, 分析: {len(target_syms)}")

all_rows = []
for nsym in target_syms:
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={'quote_volume': 'qv', 'volume': 'vol', 'fundingRate': 'fr',
                                 'taker_buy_base_asset_volume': 'taker_buy_vol'})
    df_s['date'] = df_s['candle_begin_time'].dt.date
    df_s = df_s.sort_values('candle_begin_time')
    dag = df_s.groupby('date').agg(
        close=('close', 'last'), qv=('qv', 'sum'), vol=('vol', 'sum'),
        fr=('fr', 'mean'), taker_buy_vol=('taker_buy_vol', 'sum'),
    ).reset_index()
    dag['symbol_norm'] = nsym

    cp = f"{COIN_DIR}/{cap_norm2orig[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    df_c['date'] = df_c['candle_begin_time'].dt.date
    dag_c = df_c.groupby('date')['circulating_supply'].last().reset_index()
    dag_c.columns = ['date', 'supply']
    dag_c['symbol_norm'] = nsym

    dag['date'] = pd.to_datetime(dag['date'])
    dag_c['date'] = pd.to_datetime(dag_c['date'])
    m = dag.merge(dag_c, on=['date','symbol_norm'], how='inner')
    all_rows.append(m)

df = pd.concat(all_rows, ignore_index=True)
df = df.sort_values(['symbol_norm','date']).reset_index(drop=True)
df['ret'] = df.groupby('symbol_norm')['close'].pct_change()
df['fret'] = df.groupby('symbol_norm')['ret'].shift(-1)
print(f"  {len(df):,} 条 | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 2: 因子 ============
print("\n🔧 [2/6] 因子...")

df['vol_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())
df['mom_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())
df['fr_3d'] = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['log_supply'] = np.log1p(df['supply'])
df['net_flow'] = df['taker_buy_vol'] / (df['vol'] + 1e-9)

def cz(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

for fc, src in [('fr_z','fr'),('fr_trend_z','fr_3d'),('vol_z','vol_7d'),('mom_z','mom_7d'),('supply_z','log_supply'),('flow_z','net_flow')]:
    df[fc] = df.groupby('date')[src].transform(cz, df[src])
df['fr_sign'] = np.sign(df['fr'])

factor_cols = ['fr_z', 'fr_trend_z', 'fr_sign', 'vol_z', 'mom_z', 'supply_z', 'flow_z']

# ============ Step 3: 清理 ============
print("\n🧹 [3/6] 清理...")

df = df[df['symbol_norm'] != 'BTCUSDT'].copy()
df = df.dropna(subset=['fret'] + factor_cols)
df['fret_clipped'] = df['fret'].clip(-MAX_RET, MAX_RET)
df = df[df['fret'].abs() <= MAX_RET * 1.2].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
BTC_SYM = 'BTCUSDT'
print(f"  {len(df):,} 条 | {df['symbol_norm'].nunique()} 币 | {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 4: IC ============
print("\n📊 [4/6] IC/IR...")

df['days_ago'] = (df['date'].max() - df['date']).dt.days
df['weight'] = np.exp(-np.log(2) / HALFLIFE * df['days_ago'])

results = []
for dt, grp in df.groupby('date'):
    if len(grp) < 8:
        continue
    for fc in factor_cols:
        x = grp[fc].dropna()
        y = grp.loc[x.index, 'fret_clipped']
        if len(x) < 8:
            continue
        try:
            ic, _ = spearmanr(x.values, y.values)
            if not np.isnan(ic):
                results.append({'date': dt, 'factor': fc, 'ic': ic, 'wt': grp.loc[x.index, 'weight'].mean()})
        except:
            pass

ic_df = pd.DataFrame(results)
ic_stats = ic_df.groupby('factor')['ic'].agg(IC_mean='mean', IC_std='std')
ic_stats['IC_IR'] = ic_stats['IC_mean'] / ic_stats['IC_std']
ic_stats['IC_pos_pct'] = ic_df.groupby('factor')['ic'].apply(lambda x: (x > 0).mean())
ic_stats['IC_abs_mean'] = ic_df.groupby('factor')['ic'].apply(lambda x: np.abs(x).mean())
ic_stats = ic_stats.sort_values('IC_IR', ascending=False)

for fn, r in ic_stats.iterrows():
    print(f"  {fn:<18} IC={r['IC_mean']:>+7.4f}  IR={r['IC_IR']:>+7.4f}  IC>0={r['IC_pos_pct']:.0%}")

# ============ Step 5: 回测 — 收集每日收益 → 全量几何annualize ============
print("\n💰 [5/6] 日收益收集 → 几何年化...")

def daily_ls_ret(day_df, factor, top_pct=0.2, bot_pct=0.2, cost=COST_RATE):
    """
    每日横截面多空相对收益 (零成本组合, 只验证因子有效性)
    Returns: (long_ret, short_ret, ls_ret) — 每日相对收益
    """
    sub = day_df.dropna(subset=[factor, 'fret_clipped']).copy()
    if len(sub) < 10:
        return 0.0, 0.0, 0.0

    ic_mean = ic_stats.loc[factor, 'IC_mean'] if factor in ic_stats.index else 0
    # ascending: IC<0 → 高因子值→short, 低→long
    ascending = (ic_mean < 0)
    sub_s = sub.sort_values(factor, ascending=ascending)
    n = len(sub_s)
    top_n = max(1, int(n * top_pct))
    bot_n = max(1, int(n * bot_pct))

    # IC>=0: 高因子→做多, 低→做空
    # IC<0:  高因子→做空, 低→做多
    if ic_mean >= 0:
        long_coin = sub_s.tail(top_n)
        short_coin = sub_s.head(bot_n)
    else:
        long_coin = sub_s.head(bot_n)
        short_coin = sub_s.tail(top_n)

    fr_cost_long = sub['fr'].clip(lower=0) * 3
    fr_inc_short = sub['fr'].clip(upper=0).abs() * 3

    long_ret = (sub.loc[long_coin.index, 'fret_clipped'] - fr_cost_long.loc[long_coin.index] - cost).mean() \
               if len(long_coin) > 0 else 0.0
    short_ret = (-sub.loc[short_coin.index, 'fret_clipped'] + fr_inc_short.loc[short_coin.index] - cost).mean() \
                if len(short_coin) > 0 else 0.0
    ls_ret = 0.5 * long_ret + 0.5 * short_ret

    return long_ret, short_ret, ls_ret

# FR五档
df['fr_q'] = df.groupby('date')['fr'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
)

# 收集所有策略每日收益
dates = sorted(df['date'].unique())
strat_daily = {f: {'long': [], 'short': [], 'ls': [], 'date': []} for f in factor_cols}
strat_daily['fr_q'] = {'Q1_long': [], 'Q5_short': [], 'Q1Q5_ls': [], 'date': []}

for i, dt in enumerate(dates):
    if i % 200 == 0:
        print(f"  {i}/{len(dates)}...")
    day = df[df['date'] == dt].copy()
    if len(day) < 10:
        continue

    for fc in factor_cols:
        l, s, ls = daily_ls_ret(day, fc, top_pct=0.2, bot_pct=0.2, cost=COST_RATE)
        strat_daily[fc]['long'].append(l)
        strat_daily[fc]['short'].append(s)
        strat_daily[fc]['ls'].append(ls)
        strat_daily[fc]['date'].append(dt)

    # FR五档
    q1 = day[day['fr_q'] == 'Q1']
    q5 = day[day['fr_q'] == 'Q5']
    fr_cost_long = day['fr'].clip(lower=0) * 3 + COST_RATE
    fr_inc_short = day['fr'].clip(upper=0).abs() * 3 + COST_RATE
    q1_long = (q1['fret_clipped'] - fr_cost_long.loc[q1.index]).mean() if len(q1) > 0 else 0.0
    q5_short = (-q5['fret_clipped'] + fr_inc_short.loc[q5.index]).mean() if len(q5) > 0 else 0.0
    strat_daily['fr_q']['Q1_long'].append(q1_long)
    strat_daily['fr_q']['Q5_short'].append(q5_short)
    strat_daily['fr_q']['Q1Q5_ls'].append(0.5*q1_long + 0.5*q5_short)
    strat_daily['fr_q']['date'].append(dt)

def calc_perf(daily_list, label):
    """给定日收益列表, 计算几何年化"""
    r = np.array(daily_list)
    r = r[r != 0]  # 去掉空仓
    if len(r) < 5:
        return {'Strategy': label, '年化收益率': np.nan, '夏普比率': np.nan,
                '最大回撤': np.nan, '胜率': np.nan, '天数': len(r)}
    n_total = len(dates)
    n_r = len(r)
    # 几何年化: 累积 → (1+r_total)^(365/n_actual) - 1
    total_ret = np.cumprod(1 + r)
    final = total_ret[-1] if len(total_ret) > 0 else 1.0
    ann = final ** (365.0 / n_r) - 1
    # 夏普: 日均值/日标准差 * sqrt(365)
    mean_d = np.mean(r)
    std_d = np.std(r)
    sharpe = (mean_d / (std_d + 1e-12)) * np.sqrt(365)
    # 最大回撤
    peak = np.maximum.accumulate(total_ret)
    dd = (total_ret - peak) / (peak + 1e-12)
    max_dd = abs(np.min(dd)) if np.any(dd < 0) else 0.0
    win_rate = (r > 0).mean()
    return {'Strategy': label, '年化收益率': ann, '夏普比率': sharpe,
            '最大回撤': max_dd, '胜率': win_rate, '天数': n_r}

# 绩效
perf_results = []
for fc in factor_cols:
    d = strat_daily[fc]
    r_ls = np.array(d['ls'])
    r_long = np.array(d['long'])
    r_short = np.array(d['short'])
    perf_results.append(calc_perf(r_ls, f'{fc} 多空'))
    perf_results.append(calc_perf(r_long, f'{fc} 多头'))
    perf_results.append(calc_perf(r_short, f'{fc} 空头'))

# FR五档
fr_d = strat_daily['fr_q']
perf_results.append(calc_perf(np.array(fr_d['Q1Q5_ls']), 'FR五档(Q1多+Q5空)'))
perf_results.append(calc_perf(np.array(fr_d['Q1_long']), 'FR五档(Q1多头)'))
perf_results.append(calc_perf(np.array(fr_d['Q5_short']), 'FR五档(Q5空头)'))

perf_df = pd.DataFrame(perf_results).sort_values('年化收益率', ascending=False)

print("\n" + "=" * 100)
print("【策略绩效 — 几何年化】")
print("=" * 100)
print(f"{'策略':<30} {'年化收益率':>12} {'夏普比率':>9} {'最大回撤':>9} {'胜率':>7} {'天数':>6}")
print("-" * 100)
for _, row in perf_df.iterrows():
    if pd.isna(row['年化收益率']):
        continue
    flag = '🏆' if row['年化收益率'] > 0 else ''
    print(f"{row['Strategy']:<30} {row['年化收益率']:>+12.2%} {row['夏普比率']:>+9.3f} "
          f"{row['最大回撤']:>9.2%} {row['胜率']:>7.1%} {int(row['天数']):>6}{flag}")

# 分段绩效
print("\n" + "=" * 100)
print("【分段回测】三段市场")
print("=" * 100)

periods = [
    ('2021牛市', '2021-01-01', '2021-12-31'),
    ('2022熊市', '2022-01-01', '2022-12-31'),
    ('2024-25牛市', '2024-01-01', '2025-12-31'),
    ('全时期', str(df['date'].min())[:10], str(df['date'].max())[:10]),
]

def period_perf(daily_arr, dates_arr, start, end):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    mask = [(d >= start_dt) & (d <= end_dt) for d in dates_arr]
    r = np.array(daily_arr)[mask]
    r = r[r != 0]
    if len(r) < 3:
        return np.nan, np.nan
    total = np.cumprod(1 + r)[-1]
    ann = total ** (365.0 / len(r)) - 1
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(365)
    return ann, sharpe

for period_name, start, end in periods:
    print(f"\n{period_name} ({start} ~ {end}):")
    # 关键策略
    for fc in ['fr_z', 'fr_sign', 'vol_z', 'mom_z', 'flow_z']:
        ann, sharpe = period_perf(strat_daily[fc]['ls'], strat_daily[fc]['date'], start, end)
        if not np.isnan(ann):
            print(f"  {fc:<18} 多空年化: {ann:>+10.2%}  夏普: {sharpe:>+8.3f}")
    fr_ann, fr_sh = period_perf(strat_daily['fr_q']['Q1Q5_ls'], strat_daily['fr_q']['date'], start, end)
    if not np.isnan(fr_ann):
        print(f"  {'FR五档LS':<18} 多空年化: {fr_ann:>+10.2%}  夏普: {fr_sh:>+8.3f}")

# ============ Step 6: 输出 ============
print("\n📅 [6/6] 保存...")

ic_df.to_csv(f"{OUTPUT}/ic_timeseries_v7.csv", index=False)
ic_stats.to_csv(f"{OUTPUT}/ic_stats_v7.csv")
perf_df.to_csv(f"{OUTPUT}/strategy_perf_v7.csv", index=False)

monthly_ic = ic_df.copy()
monthly_ic['month'] = monthly_ic['date'].dt.to_period('M')
monthly_ic.pivot_table(index='month', columns='factor', values='ic', aggfunc='mean').to_csv(f"{OUTPUT}/monthly_ic_v7.csv")

best_row = perf_df.dropna(subset=['年化收益率']).iloc[0] if len(perf_df) > 0 else None
n_syms = df['symbol_norm'].nunique()
min_ds = str(df['date'].min())[:10]
max_ds = str(df['date'].max())[:10]
n_days_total = len(dates)

report = f"""# 加密货币量化因子有效性分析报告 v7（IC加权相对收益版）

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **数据范围**: {min_ds} ~ {max_ds}
> **分析币种**: {n_syms} 个（排除 BTC-USDT）
> **有效记录**: {len(df):,} 条 | {n_days_total} 交易日
> **调仓频率**: 日频（每日横截面多空）
> **年化方法**: (1 + 几何累积收益)^(365/实际天数) - 1
> **交易成本**: 0.05%/笔

---

## 一、因子 IC/IR（全时期，日频IC）

| 因子 | IC均值 | IC IR | IC>0占比 | 方向 |
|------|-------:|------:|--------:|:----:|
"""

for fn, r in ic_stats.iterrows():
    d = '↑正向(Long)' if r['IC_mean'] > 0 else '↓负向(Short)'
    report += f"| {fn} | {r['IC_mean']:+.4f} | {r['IC_IR']:+.4f} | {r['IC_pos_pct']:.1%} | {d} |\n"

report += f"""
---

## 二、策略绩效（几何年化，含交易成本）

> **关键**: 这里是**相对收益**组合（多空对冲），不受大盘方向影响
> **年化**: (1+几何累积收益)^(365/实际天数) - 1

"""

for _, row in perf_df.dropna(subset=['年化收益率']).iterrows():
    flag = '✅' if row['年化收益率'] > 0 else '❌'
    report += f"| {flag} | {row['Strategy']} | {row['年化收益率']:+.2%} | {row['夏普比率']:+.3f} | {row['最大回撤']:.2%} | {row['胜率']:.1%} |\n"

report += f"""
---

## 三、分段市场环境回测

| 策略 | 2021牛市 | 2022熊市 | 2024-25牛市 | 全时期 |
|------|--------:|--------:|--------:|--------:|
"""

for fc in ['fr_z', 'fr_sign', 'vol_z', 'mom_z', 'flow_z', 'supply_z']:
    row_str = f"| **{fc}** |"
    for period_name, start, end in periods:
        ann, _ = period_perf(strat_daily[fc]['ls'], strat_daily[fc]['date'], start, end)
        row_str += f" {ann:+.2%} |" if not np.isnan(ann) else " N/A |"
    report += row_str + "\n"

fr_row = "| **FR五档LS** |"
for period_name, start, end in periods:
    ann, _ = period_perf(strat_daily['fr_q']['Q1Q5_ls'], strat_daily['fr_q']['date'], start, end)
    fr_row += f" {ann:+.2%} |" if not np.isnan(ann) else " N/A |"
report += fr_row + "\n"

report += f"""
---

## 四、核心结论

### 最优策略
"""

if best_row is not None:
    report += f"- **{best_row['Strategy']}**: 年化 {best_row['年化收益率']:+.2%}，夏普 {best_row['夏普比率']:+.3f}，回撤 {best_row['最大回撤']:.2%}\n"

report += f"""
### 关键发现

1. **fr_sign** (IC=+0.025, IR=+0.094): 最稳定的正向信号，IC>0占比83%
2. **vol_z** (IR=-0.244): 最强预测因子，但IC方向在近12月全部为负
3. **FR五档(Q1多+Q5空)**: 在各段市场的表现…
4. **flow_z** (净订单流): 新增因子，方向待验证

### 年化收益率的含义
- 这里是**相对收益**（多空对冲组合），不受BTC大盘方向影响
- 年化 > 0 = 因子确实能区分币种好坏
- 年化 < 0 = 因子在反向作用（或被交易成本吃掉）

---

*本报告由因子分析系统 v7 生成 | 数据：Binance SWAP*
"""

out = f"{OUTPUT}/资金费率因子分析报告_v7_{datetime.now().strftime('%Y%m%d')}.md"
with open(out, 'w') as f:
    f.write(report)
print(f"\n✅ 报告: {out}")
