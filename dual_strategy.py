# dual_strategy.py
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests
import akshare as ak
import streamlit as st
import re
import time

TX_BATCH_SIZE = 20
TX_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class DualStrategyConfig:
    # 量比触发区间（互斥，区间之间的股票不触发任何策略）
    vol_ratio_a_low: float = 0.5
    vol_ratio_a_high: float = 1.2
    vol_ratio_b_low: float = 1.5
    vol_ratio_b_high: float = 3.0

    # 策略 A 条件
    a_turnover_low: float = 1.0       # 换手率下限 %
    a_turnover_high: float = 4.0
    a_amount_min: float = 5000e4      # 成交额下限，元（5000万）
    a_mktcap_low: float = 30e8        # 总市值下限，元（30亿）
    a_mktcap_high: float = 300e8
    a_pct_low: float = -3.0           # 涨跌幅下限 %
    a_pct_high: float = 3.0
    a_below_ma60: bool = True         # True=价格在MA60下方

    # 策略 B 条件
    b_turnover_low: float = 3.0
    b_turnover_high: float = 10.0
    b_amount_min: float = 2e8         # 成交额下限，元（2亿）
    b_mktcap_low: float = 50e8
    b_mktcap_high: float = 500e8
    b_pct_low: float = 2.0
    b_pct_high: float = 5.0
    b_above_ma20: bool = True         # True=价格在MA20上方


# ---------- 腾讯批量行情 ----------
def _code_to_tx(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    elif code.startswith(("0", "2", "3")):
        return f"sz{code}"
    return f"bj{code}"


def _parse_tx_batch(raw_text: str) -> list[dict]:
    rows = []
    # 匹配 v_sh600519="..." 或 v_sz000001="..."，不依赖 ; 或 \n 分隔
    for m in re.finditer(r'v_([a-z]{2}\d+)="([^"]*)"', raw_text):
        code_full, payload = m.group(1), m.group(2)
        parts = payload.split("~")
        if len(parts) < 48:
            continue
        try:
            rows.append({
                "code": parts[2].zfill(6),
                "name": parts[1],
                "price": float(parts[3]),
                "pct_chg": float(parts[32]),
                "turnover": float(parts[38]),
                "amount": float(parts[37]) * 1e4,
                "total_mktcap": float(parts[45]) * 1e8,
                "vol_ratio": float(parts[47]),
            })
        except (ValueError, IndexError):
            continue
    return rows


@st.cache_data(ttl=60)
def fetch_snapshot_all() -> pd.DataFrame:
    code_df = ak.stock_info_a_code_name()
    all_codes = code_df["code"].astype(str).str.zfill(6).tolist()

    all_rows = []
    for i in range(0, len(all_codes), TX_BATCH_SIZE):
        batch = all_codes[i:i + TX_BATCH_SIZE]
        tx_codes = ",".join(_code_to_tx(c) for c in batch)
        try:
            r = requests.get(
            f"https://qt.gtimg.cn/q={tx_codes}",
            headers=TX_HEADERS, timeout=10,
            )            
            r.encoding = "gbk"
            all_rows.extend(_parse_tx_batch(r.text))
        except Exception:
            continue
        time.sleep(0.05)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    # 排除停牌（价格为0）和ST
    df = df[df["price"] > 0]
    df = df[~df["name"].str.contains("ST", na=False)]
    return df.reset_index(drop=True)


# ---------- 历史日线 + MA ----------
@st.cache_data(ttl=3600, show_spinner=False)
def _hist_for_ma(code: str, days: int = 120) -> pd.DataFrame:
    """拉单只股票近 days 天前复权日线，用于算 MA。"""
    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=code, adjust="qfq",
            start_date=(pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d"),
            end_date=pd.Timestamp.today().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"收盘": "close"})[["close"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


def attach_ma(df: pd.DataFrame, ma_windows=(20, 60)) -> pd.DataFrame:
    """对候选池逐只拉日线，计算最新 MA20/MA60，合并回快照。"""
    ma_records = []
    for _, row in df.iterrows():
        hist = _hist_for_ma(row["code"])
        if hist.empty or len(hist) < max(ma_windows):
            ma_records.append({"code": row["code"], "ma20": np.nan, "ma60": np.nan})
            continue
        closes = hist["close"]
        ma_records.append({
            "code": row["code"],
            "ma20": round(closes.rolling(20).mean().iloc[-1], 2) if 20 in ma_windows else np.nan,
            "ma60": round(closes.rolling(60).mean().iloc[-1], 2) if 60 in ma_windows else np.nan,
        })
    return df.merge(pd.DataFrame(ma_records), on="code", how="left")


# ---------- 双策略筛选 ----------
def _apply_strategy(
    df: pd.DataFrame, cfg: DualStrategyConfig, which: str
) -> pd.DataFrame:
    """which='A' 或 'B'，返回满足该策略全部条件的子集。"""
    if which == "A":
        vol_lo, vol_hi = cfg.vol_ratio_a_low, cfg.vol_ratio_a_high
        turn_lo, turn_hi = cfg.a_turnover_low, cfg.a_turnover_high
        amt_min = cfg.a_amount_min
        cap_lo, cap_hi = cfg.a_mktcap_low, cfg.a_mktcap_high
        pct_lo, pct_hi = cfg.a_pct_low, cfg.a_pct_high
        ma_col, ma_side = "ma60", "below"
    else:
        vol_lo, vol_hi = cfg.vol_ratio_b_low, cfg.vol_ratio_b_high
        turn_lo, turn_hi = cfg.b_turnover_low, cfg.b_turnover_high
        amt_min = cfg.b_amount_min
        cap_lo, cap_hi = cfg.b_mktcap_low, cfg.b_mktcap_high
        pct_lo, pct_hi = cfg.b_pct_low, cfg.b_pct_high
        ma_col, ma_side = "ma20", "above"

    mask = df["vol_ratio"].between(vol_lo, vol_hi)
    mask &= df["turnover"].between(turn_lo, turn_hi)
    mask &= df["amount"] >= amt_min
    mask &= df["total_mktcap"].between(cap_lo, cap_hi)
    mask &= df["pct_chg"].between(pct_lo, pct_hi)

    if ma_col in df.columns:
        if ma_side == "below":
            mask &= df["price"] < df[ma_col]
        else:
            mask &= df["price"] > df[ma_col]

    return df[mask.fillna(False)].reset_index(drop=True)


def run_dual_strategy(cfg: DualStrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (策略A结果, 策略B结果)。
    实时快照做初筛，只对初筛后的候选池算 MA，避免全市场遍历历史。
    """
    snap = fetch_snapshot_all()

    # 初筛（不含 MA），大幅缩小候选池
    pre_a = snap[
        snap["vol_ratio"].between(cfg.vol_ratio_a_low, cfg.vol_ratio_a_high)
        & snap["turnover"].between(cfg.a_turnover_low, cfg.a_turnover_high)
        & (snap["amount"] >= cfg.a_amount_min)
        & snap["total_mktcap"].between(cfg.a_mktcap_low, cfg.a_mktcap_high)
        & snap["pct_chg"].between(cfg.a_pct_low, cfg.a_pct_high)
    ].copy()

    pre_b = snap[
        snap["vol_ratio"].between(cfg.vol_ratio_b_low, cfg.vol_ratio_b_high)
        & snap["turnover"].between(cfg.b_turnover_low, cfg.b_turnover_high)
        & (snap["amount"] >= cfg.b_amount_min)
        & snap["total_mktcap"].between(cfg.b_mktcap_low, cfg.b_mktcap_high)
        & snap["pct_chg"].between(cfg.b_pct_low, cfg.b_pct_high)
    ].copy()

    # 对两个候选池的并集算 MA（去重，避免同一只算两次）
    candidates = pd.concat([pre_a, pre_b]).drop_duplicates(subset="code")
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    candidates = attach_ma(candidates)

    # 二次过滤（带 MA 条件）
    result_a = _apply_strategy(candidates, cfg, "A")
    result_b = _apply_strategy(candidates, cfg, "B")
    return result_a, result_b


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """把内部单位转成用户可读单位。"""
    if df.empty:
        return df
    out = df.copy()
    out["成交额(万元)"] = (out["amount"] / 1e4).round(0)
    out["总市值(亿元)"] = (out["total_mktcap"] / 1e8).round(1)
    out["换手率(%)"] = out["turnover"].round(2)
    out["涨跌幅(%)"] = out["pct_chg"].round(2)
    out["量比"] = out["vol_ratio"].round(2)
    out["最新价"] = out["price"].round(2)
    out = out.rename(columns={"code": "代码", "name": "名称", "ma20": "MA20", "ma60": "MA60"})
    keep = ["代码", "名称", "最新价", "涨跌幅(%)", "换手率(%)", "量比",
            "成交额(万元)", "总市值(亿元)", "MA20", "MA60"]
    return out[[c for c in keep if c in out.columns]]
