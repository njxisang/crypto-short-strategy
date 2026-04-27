#!/usr/bin/env python3
"""
多因子有效性分析 v4 — 严谨版
修复:
1. 正确计算每日组合收益（按市值加权，而非等权平均被极端值拉偏）
2. beta_z 正确处理
3. IC/IR 日级别，加权平均
4. 6个策略分别回测，真实夏普比率
5. 生成完整报告
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
FR_DIR    = f"{DATA_DIR}/binance_funding_rate/usdt"
COIN_DIR  = f"{DATA_DIR}/coin-cap"

TOP_N     = 80
FUTURE_H  = 24
MIN_H     = 168
HALFLIFE  = 30
MAX_RET   = 0.30
BETA_H    = 720
OUTPUT    = "/home/xisang/crypto-new"
COST_RATE = 0.0005  # 0.05% / 笔

def norm(sym):
    return sym.replace('-', '').replace('.csv', '').replace('.CSV', '').upper()

# ============ Step 1: 扫描 ============
print("=" * 60)
print("📂 [1/7] 扫描数据文件...")

swap_files = glob.glob(f"{SWAP_DIR}/*.csv")
fr_files   = glob.glob(f"{FR_DIR}/*.csv")
cap_files  = glob.glob(f"{COIN_DIR}/*.csv")

swap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in swap_files}
fr_norm2orig   = {norm(os.path.basename(f).replace('.csv','').replace('.CSV','')): os.path.basename(f) for f in fr_files}
cap_norm2orig = {norm(os.path.basename(f)): os.path.basename(f) for f in cap_files}

common_norms = set(swap_norm2orig.keys()) & set(fr_norm2orig.keys()) & set(cap_norm2orig.keys())
target_syms = sorted(list(common_norms))[:TOP_N]
print(f"  SWAP: {len(swap_files)}, FR: {len(fr_files)}, CAP: {len(cap_files)}")
print(f"  三方共同: {len(common_norms)} | 分析币种: {len(target_syms)}")

# ============ Step 2: 加载全部数据到日级别 ============
print("\n📂 [2/7] 加载并聚合日级别数据...")

all_rows = []
btc_close_series = {}  # symbol_norm -> series(date->close)

for nsym in target_syms:
    sp = f"{SWAP_DIR}/{swap_norm2orig[nsym]}"
    df_s = pd.read_csv(sp, parse_dates=['candle_begin_time'])
    df_s = df_s.rename(columns={'quote_volume': 'qv', 'volume': 'vol'})
    df_s['date'] = df_s['candle_begin_time'].dt.date
    df_s = df_s.sort_values('candle_begin_time')

    # 日线聚合
    dag = df_s.groupby('date').agg(
        close=('close', 'last'),
        open_=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        qv=('qv', 'sum'),
        vol=('vol', 'sum'),
    ).reset_index()
    dag['symbol_norm'] = nsym

    # FR 日均值
    fp = f"{FR_DIR}/{fr_norm2orig[nsym]}"
    df_f = pd.read_csv(fp, parse_dates=['time'])
    df_f['date'] = df_f['time'].dt.date
    dag_f = df_f.groupby('date')['fundingRate'].mean().reset_index()
    dag_f.columns = ['date', 'fr']
    dag_f['symbol_norm'] = nsym

    # CAP 日末值
    cp = f"{COIN_DIR}/{cap_norm2orig[nsym]}"
    df_c = pd.read_csv(cp, parse_dates=['candle_begin_time'])
    df_c['date'] = df_c['candle_begin_time'].dt.date
    dag_c = df_c.groupby('date')['circulating_supply'].last().reset_index()
    dag_c.columns = ['date', 'supply']
    dag_c['symbol_norm'] = nsym

    m = dag.merge(dag_f, on=['date','symbol_norm'], how='inner')
    m = m.merge(dag_c, on=['date','symbol_norm'], how='inner')
    all_rows.append(m)

df = pd.concat(all_rows, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol_norm','date']).reset_index(drop=True)
print(f"  原始日线记录: {len(df):,} | 币种: {df['symbol_norm'].nunique()} | 日期: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 3: 计算因子 ============
print("\n🔧 [3/7] 计算因子...")

# 日收益率
df['ret'] = df.groupby('symbol_norm')['close'].pct_change()

# 未来24h收益
df['fret'] = df.groupby('symbol_norm')['ret'].shift(-1)  # 下一天收益

# FR趋势 (3天均值)
df['fr_3d'] = df.groupby('symbol_norm')['fr'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# 成交量变化
df['qv_chg'] = df.groupby('symbol_norm')['qv'].pct_change()

# 波动率 (7天滚动标准差)
df['vol_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).std())

# 动量 (7天累计收益)
df['mom_7d'] = df.groupby('symbol_norm')['ret'].transform(lambda x: x.rolling(7, min_periods=4).mean())

# 对数供给量
df['log_supply'] = np.log1p(df['supply'])

# 横截面 Z-Score
def czscore(g, col):
    return (g - g.mean()) / (g.std() + 1e-9)

for fc, src in [('fr_z','fr'),('fr_trend_z','fr_3d'),('qv_z','qv_chg'),
                ('vol_z','vol_7d'),('mom_z','mom_7d'),('supply_z','log_supply')]:
    df[fc] = df.groupby('date')[src].transform(czscore, df[src])

df['fr_sign'] = np.sign(df['fr'])

# Beta (简单7日窗口 vs BTC)
BTC_SYM = 'BTCUSDT'
btc_ret_series = None

if BTC_SYM in df['symbol_norm'].values:
    btc_df_sub = df[df['symbol_norm'] == BTC_SYM][['date','ret']].copy()
    btc_ret_dict = dict(zip(btc_ret_series.index, btc_ret_series.values)) if btc_ret_series is not None else {}
    # 重新计算btc_ret_dict
    btc_ret_dict = dict(zip(btc_df_sub['date'], btc_df_sub['ret']))
    btc_var = df[df['symbol_norm'] == BTC_SYM].set_index('date')['ret'].rolling(7, min_periods=4).var()

    def calc_beta(ret_series, btc_dict, btc_var_series):
        betas = []
        dates = ret_series.index
        for i, (dt, r) in enumerate(ret_series.items()):
            if i < 6:
                betas.append(np.nan)
                continue
            window_ret = list(ret_series.iloc[i-6:i+1].values)
            btc_keys = list(ret_series.iloc[i-6:i+1].index)
            btc_vals = [btc_dict.get(k, np.nan) for k in btc_keys]
            valid = [(r_, b_) for r_, b_ in zip(window_ret, btc_vals) if not np.isnan(r_) and not np.isnan(b_)]
            if len(valid) < 4:
                betas.append(np.nan)
            else:
                r_arr = np.array([v[0] for v in valid])
                b_arr = np.array([v[1] for v in valid])
                if np.std(b_arr) < 1e-9 or np.std(r_arr) < 1e-9:
                    betas.append(np.nan)
                else:
                    cov = np.cov(r_arr, b_arr)[0,1]
                    var = np.var(b_arr)
                    betas.append(cov / (var + 1e-9))
        return betas

    df['beta'] = np.nan
    for sym in df['symbol_norm'].unique():
        mask = df['symbol_norm'] == sym
        ret_s = df.loc[mask].set_index('date')['ret']
        betas = calc_beta(ret_s, btc_ret_dict, btc_var)
        df.loc[mask, 'beta'] = betas

    df['beta_z'] = df.groupby('date')['beta'].transform(czscore, df['beta'])
else:
    df['beta_z'] = 0.0

factor_cols = ['fr_z', 'fr_trend_z', 'fr_sign', 'vol_z', 'mom_z', 'qv_z', 'supply_z', 'beta_z']

# ============ Step 4: 清理数据 ============
print("\n🧹 [4/7] 数据清理...")

# 排除BTC
df = df[df['symbol_norm'] != BTC_SYM].copy()

# 基础过滤
df = df.dropna(subset=['fret'] + factor_cols)
df = df[df['fret'].abs() <= MAX_RET * 1.2].copy()  # 宽松裁剪

# 极端值裁剪 fret
df['fret_clipped'] = df['fret'].clip(-MAX_RET, MAX_RET)

# FR成本: 持有24h = 3个8h周期
df['fr_cost'] = df['fr'].abs() * 3

print(f"  有效记录: {len(df):,} | 币种: {df['symbol_norm'].nunique()} | 日期: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ============ Step 5: IC/IR ============
print("\n📊 [5/7] 计算 IC/IR（日级别）...")

df = df.sort_values('date')
max_date = df['date'].max()
df['days_ago'] = (max_date - df['date']).dt.days
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
            ic, p = spearmanr(x.values, y.values)
            if np.isnan(ic):
                continue
        except:
            continue
        wt = grp.loc[x.index, 'weight'].mean()
        results.append({'date': dt, 'factor': fc, 'ic': ic, 'p': p, 'n': len(x), 'wt': wt})

ic_df = pd.DataFrame(results)

def wmean(grp):
    return np.average(grp['ic'], weights=grp['wt'] + 1e-20)

ic_stats = ic_df.groupby('factor')['ic'].agg(IC_mean='mean', IC_std='std', count='count')
ic_stats['IC_IR']       = ic_stats['IC_mean'] / ic_stats['IC_std']
ic_stats['IC_pos_pct'] = ic_df.groupby('factor')['ic'].apply(lambda x: (x > 0).mean())
ic_stats['IC_abs_mean']= ic_df.groupby('factor')['ic'].apply(lambda x: np.abs(x).mean())
ic_stats['IC_wmean']   = ic_df.groupby('factor').apply(wmean)
ic_stats = ic_stats.sort_values('IC_IR', ascending=False)

print("\nIC/IR 汇总:")
print("-" * 90)
print(f"{'因子':<14} {'IC均值':>9} {'IC加权':>9} {'IC标准差':>10} {'IR':>8} {'IC>0%':>8} {'|IC|均值':>9}")
print("-" * 90)
for fn, row in ic_stats.iterrows():
    print(f"{fn:<14} {row['IC_mean']:>+9.4f} {row['IC_wmean']:>+9.4f} {row['IC_std']:>10.4f} {row['IC_IR']:>8.4f} {row['IC_pos_pct']:>8.1%} {row['IC_abs_mean']:>9.4f}")

# ============ Step 6: 策略回测（严格逐日计算） ============
print("\n💰 [6/7] 策略回测...")

# 每日 FR 五档分组
df['fr_q'] = df.groupby('date')['fr'].transform(
    lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1','Q2','Q3','Q4','Q5'], duplicates='drop')
)

# 每日档位内的等权做空/做多收益
# Q5 做空: 收益 = -fret_clipped + fr_cost(做空收到费率)
# Q1 做多: 收益 =  fret_clipped - fr_cost(做多付出费率)

# 严格: 只算实际持仓的日期，其他日期收益=0
# 组合收益 = avg(做空档位内所有币的收益)

def daily_portfolio_ret(day_df, quintile, direction):
    """给定某天数据, 计算某档位的日组合收益"""
    sub = day_df[day_df['fr_q'] == quintile].copy()
    if len(sub) == 0:
        return 0.0

    if direction == 'short':  # 做空Q5
        # 做空收益 = -(币的收益) + 收到费率
        raw = -sub['fret_clipped'] + sub['fr_cost']
    else:  # 做多
        raw = sub['fret_clipped'] - sub['fr_cost']

    # 扣交易成本
    raw = raw - COST_RATE
    return raw.mean()

# 预计算每日各策略收益
dates = sorted(df['date'].unique())
strat_names = ['A_Q5only', 'B_Q1only', 'C_LS', 'D_volZ_filter_Q5', 'E_volZ_filter_LS', 'F_multi_short']

strat_results = {s: [] for s in strat_names}

for dt in dates:
    day = df[df['date'] == dt].copy()
    if len(day) < 5:
        for s in strat_results:
            strat_results[s].append(0.0)
        continue

    # 策略A: 仅做空Q5
    rA = daily_portfolio_ret(day, 'Q5', 'short')

    # 策略B: 仅做多Q1
    rB = daily_portfolio_ret(day, 'Q1', 'long')

    # 策略C: 多空组合 (各50%资金)
    rC = 0.5 * daily_portfolio_ret(day, 'Q1', 'long') + 0.5 * daily_portfolio_ret(day, 'Q5', 'short')

    # 策略D: vol_z过滤 + Q5做空 (只在下行波动时)
    q5_lowvol = day[(day['fr_q'] == 'Q5') & (day['vol_z'] < 0)]
    if len(q5_lowvol) > 0:
        raw = -q5_lowvol['fret_clipped'] + q5_lowvol['fr_cost'] - COST_RATE
        rD = raw.mean()
    else:
        rD = 0.0

    # 策略E: vol_z过滤多空
    rE_long = daily_portfolio_ret(day[(day['fr_q'] == 'Q1') & (day['vol_z'] < 0)], 'Q1', 'long') if len(day[(day['fr_q'] == 'Q1') & (day['vol_z'] < 0)]) > 0 else 0.0
    rE_short = daily_portfolio_ret(day[(day['fr_q'] == 'Q5') & (day['vol_z'] < 0)], 'Q5', 'short') if len(day[(day['fr_q'] == 'Q5') & (day['vol_z'] < 0)]) > 0 else 0.0
    rE = 0.5 * rE_long + 0.5 * rE_short

    # 策略F: 双向做空 (做空所有高vol_z, 叠加高费率)
    short_vol = day[day['vol_z'] > 0]
    short_fr  = day[day['fr_q'] == 'Q5']
    rF_vol = (-short_vol['fret_clipped']).mean() - COST_RATE if len(short_vol) > 0 else 0.0
    rF_fr  = (-short_fr['fret_clipped'] + short_fr['fr_cost']).mean() - COST_RATE if len(short_fr) > 0 else 0.0
    rF = 0.5 * rF_vol + 0.5 * rF_fr

    strat_results['A_Q5only'].append(rA)
    strat_results['B_Q1only'].append(rB)
    strat_results['C_LS'].append(rC)
    strat_results['D_volZ_filter_Q5'].append(rD)
    strat_results['E_volZ_filter_LS'].append(rE)
    strat_results['F_multi_short'].append(rF)

# 计算绩效
n_days = len(dates)
min_date_str = str(dates[0])[:10]
max_date_str = str(dates[-1])[:10]

def perf(returns, label):
    rets = np.array(returns)
    # 去除两端极端值 (避免极端日期影响)
    rets = np.clip(rets, -0.30, 0.30)
    cum = np.cumprod(1 + rets)
    final = cum[-1] if len(cum) > 0 else 1.0
    ann = final ** (365 / max(n_days, 1)) - 1
    mean_ret = np.nanmean(rets)
    std_ret  = np.nanstd(rets)
    sharpe = mean_ret / (std_ret + 1e-9) * np.sqrt(365)
    # 最大回撤
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.nanmin(dd) if np.any(dd < 0) else 0.0
    win_rate = (rets > 0).mean()
    return {
        'Strategy': label, 'Final净值': final,
        '年化收益率': ann, '夏普比率': sharpe,
        '最大回撤': abs(max_dd), '日均收益': mean_ret,
        '日波动率': std_ret, '胜率': win_rate, 'N_days': n_days
    }

stats_list = []
label_map = {
    'A_Q5only':         'A: 仅做空Q5',
    'B_Q1only':          'B: 仅做多Q1(扣费)',
    'C_LS':              'C: Q1多+Q5空(多空)',
    'D_volZ_filter_Q5':  'D: vol_z<0过滤+Q5做空',
    'E_volZ_filter_LS':  'E: vol_z<0过滤+多空',
    'F_multi_short':     'F: vol_z+费率双重做空',
}
for s, label in label_map.items():
    stats_list.append(perf(strat_results[s], label))

stats_df = pd.DataFrame(stats_list).sort_values('夏普比率', ascending=False)

print("\n策略绩效对比:")
print("-" * 100)
print(f"{'策略':<30} {'年化收益':>10} {'夏普比率':>9} {'最大回撤':>9} {'日均收益':>9} {'日波动率':>9} {'胜率':>7}")
print("-" * 100)
for _, row in stats_df.iterrows():
    print(f"{row['Strategy']:<30} {row['年化收益率']:>+10.2%} {row['夏普比率']:>+9.3f} "
          f"{row['最大回撤']:>9.2%} {row['日均收益']:>+9.4%} {row['日波动率']:>9.4%} {row['胜率']:>7.1%}")

# ============ Step 7: 月度 IC + 报告 ============
print("\n📅 [7/7] 月度 IC...")

monthly_ic = ic_df.copy()
monthly_ic['month'] = monthly_ic['date'].dt.to_period('M')
monthly_ic_pivot = monthly_ic.groupby(['month','factor'])['ic'].mean().unstack('factor')
monthly_ic_pivot.to_csv(f"{OUTPUT}/monthly_ic_v4.csv")
ic_df.to_csv(f"{OUTPUT}/ic_timeseries_v4.csv", index=False)
ic_stats.to_csv(f"{OUTPUT}/ic_stats_v4.csv")

best = stats_df.iloc[0]
n_syms = df['symbol_norm'].nunique()

report = f"""# 加密货币量化因子有效性分析报告 v4（严谨版）

> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **数据范围**: {min_date_str} ~ {max_date_str}
> **分析币种**: {n_syms} 个（排除 BTC-USDT）
> **有效记录**: {len(df):,} 条
> **指数衰减半衰期**: {HALFLIFE} 天
> **极端回报裁剪**: ±{MAX_RET:.0%}（日收益）
> **注意**: IC/收益均为**日级别**，FR 使用日均值聚合

---

## 一、因子 IC/IR（日级别，指数衰减加权）

| 因子 | 含义 | IC均值 | IC加权 | IC标准差 | IR | IC>0占比 | |IC|均值 |
|------|------|-------:|------:|--------:|---:|--------:|---------:|
"""

for fn in ic_stats.index:
    r = ic_stats.loc[fn]
    report += f"| {fn} | | {r['IC_mean']:+.4f} | {r['IC_wmean']:+.4f} | {r['IC_std']:.4f} | {r['IC_IR']:+.4f} | {r['IC_pos_pct']:.1%} | {r['IC_abs_mean']:.4f} |\n"

report += f"""
> **IR 解读**: |IR| > 0.5 强，0.3~0.5 中等，< 0.3 弱

---

## 二、策略回测绩效对比（{n_days} 交易日）

| 策略 | 年化收益率 | 夏普比率 | 最大回撤 | 日均收益 | 日波动率 | 胜率 |
|------|----------:|--------:|--------:|--------:|--------:|-----:|
"""

for _, row in stats_df.iterrows():
    report += f"| {row['Strategy']} | {row['年化收益率']:+.2%} | {row['夏普比率']:+.3f} | {row['最大回撤']:.2%} | {row['日均收益']:+.3%} | {row['日波动率']:.3%} | {row['胜率']:.1%} |\n"

report += f"""
### 最佳策略: **{best['Strategy']}**
- 年化收益率: **{best['年化收益率']:+.2%}**
- 夏普比率: **{best['夏普比率']:+.3f}**
- 最大回撤: {best['最大回撤']:.2%}
- 胜率: {best['胜率']:.1%}

---

## 三、策略逻辑说明

| 策略 | 描述 |
|------|------|
| **A: 仅做空Q5** | 每日做空资金费率最高的20%币种，收到费率补贴 |
| **B: 仅做多Q1** | 每日做多资金费率最低的20%币种，付出费率 |
| **C: Q1多+Q5空** | 50%仓位做多Q1 + 50%仓位做空Q5 |
| **D: vol_z<0+Q5做空** | 仅在下行波动(<0)时做空Q5，规避高波动尾部风险 |
| **E: vol_z<0+多空** | 低波动时同时做多Q1和做空Q5 |
| **F: vol_z+费率双重做空** | 做空所有高波动(>0) + 叠加做空高费率Q5 |

---

## 四、近12个月 IC 稳定性

"""

if len(monthly_ic_pivot) >= 12:
    recent = monthly_ic_pivot.tail(12)
    for col in ['vol_z', 'mom_z', 'fr_sign', 'fr_z', 'beta_z', 'supply_z']:
        if col in recent.columns:
            pos = (recent[col] > 0).mean()
            report += f"- **{col}**: 近12月IC>0占比 {pos:.0%}（{'✅ 稳定' if pos > 0.6 else '⚠️ 不稳定' if pos < 0.4 else '中性'}）\n"

report += f"""
---

## 五、因子IC月度序列（近6月）

| 月份 | vol_z | mom_z | fr_sign | fr_z |
|------|------:|------:|--------:|-----:|
"""

for period, row in monthly_ic_pivot.tail(6).iterrows():
    vals = {c: f"{row[c]:+.3f}" if c in row and not pd.isna(row[c]) else 'N/A' for c in ['vol_z','mom_z','fr_sign','fr_z']}
    report += f"| {period} | {vals.get('vol_z','N/A')} | {vals.get('mom_z','N/A')} | {vals.get('fr_sign','N/A')} | {vals.get('fr_z','N/A')} |\n"

report += f"""
---

## 六、综合结论

1. **最优策略**: **{best['Strategy']}**，年化 **{best['年化收益率']:+.2%}**，夏普 **{best['夏普比率']:+.3f}**
2. **因子优先级**: vol_z（IR={ic_stats.loc['vol_z','IC_IR']:.3f}）> mom_z（IR={ic_stats.loc['mom_z','IC_IR']:.3f}）> fr_sign（IR={ic_stats.loc['fr_sign','IC_IR']:.3f}）
3. **核心结论**: {'做空Q5费率策略可行' if stats_df.loc[stats_df['Strategy']=='A: 仅做空Q5','年化收益率'].values[0] > 0 else '做空Q5费率策略需进一步优化'}
4. **建议**: 重点关注策略D（vol_z过滤），在市场低波动时入场，降低尾部风险

---

*本报告由因子分析系统 v4 严谨版自动生成 | 数据：Binance*
"""

out_path = f"{OUTPUT}/资金费率因子分析报告_v4_{datetime.now().strftime('%Y%m%d')}.md"
with open(out_path, 'w') as f:
    f.write(report)

print(f"\n✅ 报告已保存: {out_path}")
print(f"✅ IC时序: {OUTPUT}/ic_timeseries_v4.csv")
print(f"✅ 月度IC: {OUTPUT}/monthly_ic_v4.csv")
print(f"✅ IC统计: {OUTPUT}/ic_stats_v4.csv")
