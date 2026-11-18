import time
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import akshare as ak
import easyquotation
import streamlit as st

import pytz

BJ_TZ = pytz.timezone("Asia/Shanghai")


def _now_bj() -> dt.datetime:
    """返回带北京时区的当前时间。"""
    return dt.datetime.now(pytz.utc).astimezone(BJ_TZ)

@dataclass
class TailConfig:
    float_mktcap_max: float = 30e8
    price_min: float = 2.0
    price_max: float = 50.0
    exclude_st: bool = True
    exclude_kcb: bool = True
    exclude_bj: bool = True
    sample_start: str = "14:45"
    sample_end: str = "14:55"
    sample_interval_sec: int = 30
    max_stale_sec: int = 30
    w_tail_vol_ratio: float = 0.40
    w_tail_slope: float = 0.35
    w_pct_chg: float = 0.25
    top_n: int = 10
    pct_chg_min: float = 1.0
    pct_chg_max: float = 7.0


def build_micro_pool(cfg: TailConfig = TailConfig()) -> pd.DataFrame:
    spot = ak.stock_zh_a_spot_em()
    df = spot.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)

    df = df[df["最新价"].between(cfg.price_min, cfg.price_max)]
    df = df[df["流通市值"].notna()]
    df = df[df["流通市值"] <= cfg.float_mktcap_max]
    if cfg.exclude_st:
        df = df[~df["名称"].str.contains("ST", na=False)]
    if cfg.exclude_kcb:
        df = df[~df["代码"].str.startswith("688")]
    if cfg.exclude_bj:
        df = df[~df["代码"].str.startswith(("8", "4"))]

    keep = ["代码", "名称", "最新价", "涨跌幅", "成交量", "成交额", "换手率", "流通市值", "量比"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    return df


@dataclass
class Snapshot:
    ts: dt.datetime
    data: Dict[str, dict] = field(default_factory=dict)


def _normalize_code(code: str) -> str:
    return str(code).zfill(6)


def _snapshot_arrival_time(raw: dict) -> Optional[dt.datetime]:
    d, t = raw.get("date"), raw.get("time")
    if d and t:
        try:
            return dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return None


def collect_tail_snapshots(codes, cfg=TailConfig(), status=None):
    quotation = easyquotation.use("sina")
    snapshots = []
    now_bj = _now_bj()
    today = now_bj.date()
    start_dt = dt.datetime.combine(today, dt.datetime.strptime(cfg.sample_start, "%H:%M").time())
    end_dt = dt.datetime.combine(today, dt.datetime.strptime(cfg.sample_end, "%H:%M").time())

    # 等待窗口开启
    while _now_bj() < start_dt.replace(tzinfo=BJ_TZ):
        remain = (start_dt.replace(tzinfo=BJ_TZ) - _now_bj()).total_seconds()
        if status:
            status.update(label=f"等待尾盘窗口（{cfg.sample_start}）… 剩余 {int(remain)} 秒")
        time.sleep(5)

    # 采样循环
    while _now_bj() <= end_dt.replace(tzinfo=BJ_TZ):
        loop_start = time.time()
        try:
            raw = quotation.real(codes)
        except Exception as e:
            if status:
                status.update(label=f"采样异常：{e}，继续重试…")
            time.sleep(cfg.sample_interval_sec)
            continue

        local_now = _now_bj()          # ← 关键：用北京时间
        accepted, stale = {}, 0
        for code, quote in raw.items():
            code6 = _normalize_code(code)
            data_ts = _snapshot_arrival_time(quote)
            if data_ts is not None:
                # 数据源返回的也是北京时间，两边同口径才能正确算时差
                if (local_now.replace(tzinfo=None) - data_ts).total_seconds() > cfg.max_stale_sec:
                    stale += 1
                    continue
            accepted[code6] = quote

        if accepted:
            snapshots.append(Snapshot(ts=local_now, data=accepted))
        if status:
            status.update(
                label=f"采样中… 已获得 {len(snapshots)} 次有效快照"
                + (f"（丢弃滞后 {stale} 只）" if stale else "")
            )

        elapsed = time.time() - loop_start
        time.sleep(max(0, cfg.sample_interval_sec - elapsed))

    return snapshots


def _get_turnover(q: dict) -> Optional[float]:
    v = q.get("turnover") if q.get("turnover") is not None else q.get("volume")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _get_now(q: dict) -> Optional[float]:
    try:
        return float(q.get("now"))
    except (TypeError, ValueError):
        return None


def compute_tail_factors(snapshots: List[Snapshot], pool: pd.DataFrame) -> pd.DataFrame:
    if len(snapshots) < 3:
        raise RuntimeError("有效快照少于 3 次，无法计算斜率类因子")

    first, last = snapshots[0], snapshots[-1]
    codes = set(first.data) & set(last.data)
    rows = []

    for code in codes:
        v_first = _get_turnover(first.data.get(code, {}))
        v_last = _get_turnover(last.data.get(code, {}))
        if v_first is None or v_last is None:
            continue
        tail_vol = v_last - v_first
        tail_vol_ratio = tail_vol / v_last if v_last > 0 else np.nan

        xs, ys, t0 = [], [], first.ts
        for snap in snapshots:
            q = snap.data.get(code)
            px = _get_now(q) if q else None
            if px is not None:
                xs.append((snap.ts - t0).total_seconds())
                ys.append(px)
        slope = np.polyfit(xs, ys, 1)[0] if len(xs) >= 3 else np.nan

        q_last = last.data.get(code, {})
        now_px = _get_now(q_last)
        try:
            prev_close = float(q_last.get("close"))
        except (TypeError, ValueError):
            prev_close = None
        pct_chg = (now_px / prev_close - 1) * 100 if (now_px and prev_close) else np.nan

        rows.append({"代码": code, "tail_vol_ratio": tail_vol_ratio,
                     "tail_slope": slope, "pct_chg_realtime": pct_chg})

    factor_df = pd.DataFrame(rows)
    return factor_df.merge(pool[["代码", "名称", "流通市值", "换手率"]], on="代码", how="inner")


def _winsorize_zscore(s: pd.Series) -> pd.Series:
    s = s.clip(s.quantile(0.01), s.quantile(0.99))
    mu, sd = s.mean(), s.std(ddof=0)
    return pd.Series(0.0, index=s.index) if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


def score_and_rank(factor_df: pd.DataFrame, cfg: TailConfig = TailConfig()) -> pd.DataFrame:
    df = factor_df.copy()
    df = df[df["pct_chg_realtime"].between(cfg.pct_chg_min, cfg.pct_chg_max)]
    if df.empty:
        return df
    df = df.dropna(subset=["tail_vol_ratio", "tail_slope", "pct_chg_realtime"])
    df["z_vol"] = _winsorize_zscore(df["tail_vol_ratio"])
    df["z_slope"] = _winsorize_zscore(df["tail_slope"])
    df["z_chg"] = _winsorize_zscore(df["pct_chg_realtime"])
    df["score"] = (cfg.w_tail_vol_ratio * df["z_vol"]
                   + cfg.w_tail_slope * df["z_slope"]
                   + cfg.w_pct_chg * df["z_chg"])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_tail_strategy(cfg: TailConfig = TailConfig(), status=None) -> Optional[pd.DataFrame]:
    if status:
        status.update(label="正在筛微盘池…")
    pool = build_micro_pool(cfg)
    if pool.empty:
        return None

    codes = pool["代码"].tolist()
    snapshots = collect_tail_snapshots(codes, cfg, status)
    if len(snapshots) < 3:
        raise RuntimeError(f"有效快照不足（{len(snapshots)} 次）")

    if status:
        status.update(label="正在计算尾盘因子并打分…")
    factors = compute_tail_factors(snapshots, pool)
    return score_and_rank(factors, cfg)
