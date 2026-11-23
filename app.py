import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from strategy_store import (
    list_strategies, load_strategy, save_strategy, delete_strategy,
)

# 20261118 尾盘筛选功能
import datetime as dt
import pytz

# 20261120 双策略
from dual_strategy import DualStrategyConfig, run_dual_strategy, to_display


BJ_TZ = pytz.timezone("Asia/Shanghai")
from tail_micro_strategy import TailConfig, run_tail_strategy
#


# ---------- 认证层 ----------
# 方式 A：从 secrets 读配置
config = {
    "cookie": dict(st.secrets["cookie"]),
    "credentials": {"usernames": {}},
}
for username, info in st.secrets["users"].items():
    config["credentials"]["usernames"][username] = {
        "name": info["name"],
        "password": info["password"],
    }

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

authenticator.login(location="main")  # 渲染登录表单

# 三种状态：未登录 / 密码错 / 已登录
if st.session_state.get("authentication_status") is False:
    st.error("用户名或密码错误")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("请先登录")
    st.stop()

# 认证通过，取当前用户名（策略按用户隔离）
current_user = st.session_state["username"]
current_name = st.session_state.get("name", current_user)

# ========== 2. 字段元数据（唯一需要维护的地方）==========
# scale 的含义：1 个界面单位 = scale 个原始单位
# 例如 turnover 原始是元，界面用万元，scale=1e4
FIELD_GROUPS = {
    "价格与涨跌": [
        {"key": "zxj",   "label": "最新价",     "unit": "元", "scale": 1,   "default": (None, None)},
        {"key": "zd",    "label": "涨跌额",     "unit": "元", "scale": 1,   "default": (None, None)},
        {"key": "zdf",   "label": "当日涨跌幅", "unit": "%",  "scale": 1,   "default": (None, None)},
        {"key": "zf",    "label": "振幅",       "unit": "%",  "scale": 1,   "default": (None, None)},
        {"key": "speed", "label": "涨速",       "unit": "",   "scale": 1,   "default": (None, None)},
    ],
    "多周期涨跌幅": [
        {"key": "zdf_d5",   "label": "5日涨跌幅",   "unit": "%", "scale": 1, "default": (None, None)},
        {"key": "zdf_d10",  "label": "10日涨跌幅",  "unit": "%", "scale": 1, "default": (None, None)},
        {"key": "zdf_d20",  "label": "20日涨跌幅",  "unit": "%", "scale": 1, "default": (None, None)},
        {"key": "zdf_d60",  "label": "60日涨跌幅",  "unit": "%", "scale": 1, "default": (None, None)},
        {"key": "zdf_w52",  "label": "52周涨跌幅",  "unit": "%", "scale": 1, "default": (None, None)},
        {"key": "zdf_y",    "label": "年初至今涨跌幅", "unit": "%", "scale": 1, "default": (None, None)},
    ],
    "成交与市值": [
        {"key": "volume",   "label": "成交量",     "unit": "手(主板)", "scale": 1,   "default": (None, None)},
        {"key": "turnover", "label": "成交额",     "unit": "万元",     "scale": 1, "default": (None, None)},
        {"key": "hsl",      "label": "换手率",     "unit": "%",        "scale": 1,   "default": (None, None)},
        {"key": "lb",       "label": "量比",       "unit": "倍",       "scale": 1,   "default": (None, None)},
        {"key": "ltsz",     "label": "流通市值",   "unit": "亿元",     "scale": 1,   "default": (None, None)},
        {"key": "zsz",      "label": "总市值",     "unit": "亿元",     "scale": 1,   "default": (None, None)},
    ],
    "估值": [
        {"key": "pe_ttm", "label": "市盈率 PE-TTM", "unit": "倍", "scale": 1, "default": (None, None)},
        {"key": "pn",     "label": "市净率 PB",     "unit": "倍", "scale": 1, "default": (None, None)},
    ],
    "资金流": [
        {"key": "zljlr",    "label": "主力净流入",   "unit": "万元", "scale": 1, "default": (None, None)},
        {"key": "zllr",     "label": "主力流入",     "unit": "万元", "scale": 1, "default": (None, None)},
        {"key": "zllc",     "label": "主力流出",     "unit": "万元", "scale": 1, "default": (None, None)},
        {"key": "zllr_d5",  "label": "5日主力流入",  "unit": "万元", "scale": 1, "default": (None, None)},
        {"key": "zllc_d5",  "label": "5日主力流出",  "unit": "万元", "scale": 1, "default": (None, None)},
    ],
}

TEXT_FIELDS = [
    {"key": "code", "label": "代码关键字"},
    {"key": "name", "label": "名称关键字"},
]

# 不在 FIELD_GROUPS 里、但需要展示的列
EXTRA_LABELS = {
    "code": "代码",
    "name": "名称",
    "pre_close": "昨收(元)",
    "stock_type": "证券类型",
    "state": "行情状态",
    "ma20": "MA20",
    "ma60": "MA60",
}



# ========== 3. 数据层（替换成你的实际取数）==========
@st.cache_data(ttl=180)
def load_data() -> pd.DataFrame:
    df = ak.stock_zh_a_spot_tx()
    return df
  

def add_pre_close(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zxj = pd.to_numeric(df["zxj"], errors="coerce")
    zd = pd.to_numeric(df["zd"], errors="coerce")
    pre = (zxj - zd).round(2).where(zxj > 0, np.nan)
    if "pre_close" in df.columns:
        df = df.drop(columns=["pre_close"])
    df["pre_close"] = pre
    cols = [c for c in df.columns if c != "pre_close"]
    return df[cols + ["pre_close"]]


# ========== 4. 动态渲染筛选控件 ==========
def apply_strategy_to_state(filters: dict):
    """把策略写进 session_state，供 render_filters 渲染时读取。"""
    for key, f in filters.items():
        if key in ("code", "name"):
            st.session_state[f"kw_{key}"] = f.get("keyword", "")
            continue

        st.session_state[f"en_{key}"] = f.get("enabled", False)
        if f.get("enabled"):
            st.session_state[f"lo_{key}"] = float(f.get("low") or 0.0)
            st.session_state[f"hi_{key}"] = float(f.get("high") or 0.0)


def render_filters() -> dict:
    """渲染所有字段控件，返回 {列名: {enabled, low, high, scale}}"""
    filters = {}

    for group_name, fields in FIELD_GROUPS.items():
        with st.sidebar.expander(group_name, expanded=(group_name == "价格与涨跌")):
            for f in fields:
                enabled = st.checkbox(
                    f'{f["label"]}' + (f'（{f["unit"]}）' if f["unit"] else ""),
                    key=f'en_{f["key"]}',
                )
                low, high = None, None
                if enabled:
                    c1, c2 = st.columns(2)
                    lo_default = f["default"][0] if f["default"][0] is not None else 0.0
                    hi_default = f["default"][1] if f["default"][1] is not None else 0.0
                    with c1:
                        low = st.number_input(
                            "下限", value=float(lo_default), key=f'lo_{f["key"]}'
                        )
                    with c2:
                        hi_placeholder = "上限（0=不限）"
                        high = st.number_input(
                            hi_placeholder, value=float(hi_default), key=f'hi_{f["key"]}'
                        )
                    # 0 表示该侧不限
                    if high == 0:
                        high = None
                filters[f["key"]] = {
                    "enabled": enabled,
                    "low": low,
                    "high": high,
                    "scale": f["scale"],
                }

    # 文本关键字
    with st.sidebar.expander("代码 / 名称", expanded=False):
        for t in TEXT_FIELDS:
            filters[t["key"]] = {
                "enabled": False,
                "keyword": st.text_input(t["label"], key=f'kw_{t["key"]}'),
            }
        filters["code"]["enabled"] = bool(filters["code"]["keyword"])
        filters["name"]["enabled"] = bool(filters["name"]["keyword"])

    return filters

def to_display_df(df: pd.DataFrame, cols: list | None = None, convert_units: bool = True) -> pd.DataFrame:
    """
    把原始缩写列名转成中文表头，并按 FIELD_GROUPS 的 scale 做单位换算。
    convert_units=True 时，turnover/资金流等列会换算成“万元”，与筛选输入单位一致。
    """
    if cols is None:
        cols = list(df.columns)

    out = df[cols].copy()
    rename_map = dict(EXTRA_LABELS)

    for fields in FIELD_GROUPS.values():
        for f in fields:
            key = f["key"]
            if key not in out.columns:
                continue

            # 单位换算（与筛选界面的单位保持一致）
            if convert_units and f["scale"] != 1:
                out[key] = pd.to_numeric(out[key], errors="coerce") / f["scale"]

            unit = f"({f['unit']})" if f["unit"] else ""
            rename_map[key] = f["label"] + unit

    return out.rename(columns=rename_map)
    
# ========== 5. 动态筛选 ==========
def filter_stocks(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    # 文本关键字
    for key in ("code", "name"):
        f = filters.get(key, {})
        if f.get("enabled") and f.get("keyword"):
            s = out[key].astype(str)
            mask &= s.str.contains(f["keyword"], case=False, na=False)

    # 数值区间
    for key, f in filters.items():
        if key in ("code", "name") or not f.get("enabled"):
            continue
        if key not in out.columns:
            st.warning(f"数据中缺少列：{key}，已跳过")
            continue

        # 原始值 / scale = 界面单位下的值
        s = pd.to_numeric(out[key], errors="coerce") / f["scale"]

        if f["low"] is not None:
            mask &= s >= f["low"]
        if f["high"] is not None:
            mask &= s <= f["high"]

    # 排除停牌（最新价 <= 0 或无最新价）
    if "zxj" in out.columns:
        mask &= pd.to_numeric(out["zxj"], errors="coerce") > 0

    return out[mask.fillna(False)].reset_index(drop=True)

###### 20261123 ma20/ma60
# ---------- 历史日线 + MA ----------
@st.cache_data(ttl=3600, show_spinner=False)
def _hist_for_ma(code: str, days: int = 90) -> pd.DataFrame:
    """拉单只股票近 days 天前复权日线，用于算 MA。"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily", adjust="qfq",
            start_date=(pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y%m%d"),
            end_date=pd.Timestamp.today().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"收盘": "close"})[["close"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

def cal_ma(df: pd.DataFrame, ma_windows=(20, 60)) -> pd.DataFrame:
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

# ========== 6. 界面主流程 ==========
st.title("选股筛选器")
st.sidebar.caption(f"当前用户：{current_name}")
authenticator.logout(location="sidebar")

# 20261118 尾盘筛选功能 + 量比双策略
# page = st.sidebar.radio("功能页面", ["自定义筛选", "尾盘微盘策略"], key="page_switch")
#

page = st.sidebar.radio(
    "功能页面",
    ["自定义筛选", "尾盘微盘策略", "量比双策略"],
    key="page_switch",
)

if page == "自定义筛选":
    
    # ---------- 在 render_filters() 调用之前，插入这段应用逻辑 ----------
    if "pending_filters" in st.session_state:
        pending = st.session_state.pop("pending_filters")
        # 现在控件还没创建，可以安全地写入 session_state
        for key, f in pending.items():
            if key in ("code", "name"):
                st.session_state[f"kw_{key}"] = f.get("keyword", "")
                continue
            st.session_state[f"en_{key}"] = f.get("enabled", False)
            if f.get("enabled"):
                st.session_state[f"lo_{key}"] = float(f.get("low") or 0.0)
                st.session_state[f"hi_{key}"] = float(f.get("high") or 0.0)

    # 8.1 渲染筛选控件（不加载数据）
    filters = render_filters()
    
    # 8.2 侧边栏：策略管理（新增）
    with st.sidebar.expander("💾 策略管理", expanded=False):
        st.markdown("**保存当前条件**")
        save_name = st.text_input("策略名称", key="save_strategy_name")
        if st.button("保存策略", key="btn_save_strategy"):
            if not save_name.strip():
                st.warning("请填写策略名称")
            else:
                try:
                    save_strategy(current_user, save_name.strip(), filters)
                    st.success(f"已保存：{save_name.strip()}")
                except Exception as e:
                    st.error(f"保存失败：{e}")
    
        st.divider()
    
        st.markdown("**调用已保存策略**")
        saved = list_strategies(current_user)
        if not saved:
            st.caption("暂无已保存的策略")
        else:
            selected = st.selectbox("选择策略", saved, key="load_strategy_select")
            c_load, c_del = st.columns(2)
            with c_load:
                if st.button("加载", key="btn_load_strategy"):
                    loaded = load_strategy(current_user, selected)
                    # if loaded:
                    #     apply_strategy_to_state(loaded)
                    #     st.session_state["loaded_strategy_name"] = selected
                    #     st.rerun()
                    # loaded_filters = load_strategy(current_user, selected)
                    if loaded_filters:
                        # 不要直接调用 apply_strategy_to_state
                        # 而是把它暂存到另一个 key
                        st.session_state["pending_filters"] = loaded_filters
                        st.session_state["loaded_strategy_name"] = selected
                        st.rerun()  # 立即重跑，在控件渲染前应用
            with c_del:
                if st.button("删除", key="btn_delete_strategy"):
                    try:
                        delete_strategy(current_user, selected)
                        st.success(f"已删除：{selected}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败：{e}")
    
    if "loaded_strategy_name" in st.session_state:
        st.sidebar.caption(f"当前已加载策略：{st.session_state['loaded_strategy_name']}")
    
    # 8.3 按钮行
    col_run, col_refresh = st.columns([1, 1])
    with col_run:
        run = st.button("开始筛选", type="primary")
    with col_refresh:
        refresh = st.button("强制刷新数据")
    
    if refresh:
        load_data.clear()
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_total", None)
        st.info("缓存已清除，请点击「开始筛选」重新拉取数据。")
    
    # 8.4 点击才加载数据 + 筛选
    if run:
        with st.spinner("正在加载行情数据并筛选，请稍候..."):
            df = add_pre_close(load_data())
            result_ = filter_stocks(df, filters)
            result = cal_ma(result_)
        st.session_state["last_result"] = result
        st.session_state["last_total"] = len(df)
    
    # 8.5 展示结果
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        total = st.session_state["last_total"]
        st.caption(f"数据共 {total} 只，筛选结果 {len(result)} 只")
    
        if result.empty:
            st.info("没有股票满足当前条件，请放宽筛选范围。")
        else:
            show_cols = [
                c for c in [
                    "code", "name", "zxj", "zdf", "hsl", "pe_ttm", "pn",
                    "ltsz", "turnover", "zljlr", "pre_close", "ma20", "ma60",
                ] if c in result.columns
            ]
            display_df = to_display_df(result, show_cols)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.download_button(
                "导出 CSV",
                display_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="filtered_stocks.csv",
                mime="text/csv",
            )
    else:
        st.info("设置筛选条件后，点击「开始筛选」。")

#  20261118 尾盘筛选功能

elif page == "尾盘微盘策略":
    st.title("尾盘微盘策略")
    st.caption("14:45–14:55 窗口内对微盘池多次采样，按尾盘量比、价格斜率、涨幅打分取 Top N。")

    with st.expander("策略参数", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            mktcap = st.number_input("流通市值上限（亿元）", value=30.0, step=5.0)
            price_min = st.number_input("最低价（元）", value=2.0, step=0.5)
            price_max = st.number_input("最高价（元）", value=50.0, step=5.0)
        with c2:
            top_n = st.number_input("取 Top N", value=10, step=5, min_value=1)
            pct_min = st.number_input("涨幅下限 (%)", value=1.0, step=0.5)
            pct_max = st.number_input("涨幅上限 (%)", value=7.0, step=0.5)
    
        # 新增：采样窗口时间输入
        st.markdown("**采样窗口（北京时间）**")
        c3, c4 = st.columns(2)
        with c3:
            start_time = st.time_input("开始时间", value=dt.time(14, 45), step=60, key="tail_start")
        with c4:
            end_time = st.time_input("结束时间", value=dt.time(14, 55), step=60, key="tail_end")
    
    # 时间合法性校验
    if start_time >= end_time:
        st.error("开始时间必须早于结束时间")
        st.stop()
    
    sample_start_str = start_time.strftime("%H:%M")
    sample_end_str = end_time.strftime("%H:%M")

    run_tail = st.button("运行尾盘微盘策略", type="primary")

    if run_tail:
        now_bj = dt.datetime.now(pytz.utc).astimezone(BJ_TZ)
        now_t = now_bj.time()
    
        if now_t > end_time:
            st.error(f"当前北京时间 {now_t:%H:%M} 已超过采样结束时间 {sample_end_str}")
        else:
            # 窗口内才点 → 明确警告会占用采样时间
            if start_time <= now_t <= end_time:
                st.warning(
                    f"当前已在采样窗口内（{sample_start_str}–{sample_end_str}）。"
                    "筛池约需 30–60 秒，会占用采样时间。下次建议在窗口前点击运行，"
                    "让筛池和等待都在窗口外完成。"
                )
    
            cfg = TailConfig(
                float_mktcap_max=mktcap * 1e8,
                price_min=price_min, price_max=price_max,
                top_n=int(top_n),
                pct_chg_min=pct_min, pct_chg_max=pct_max,
                sample_start=sample_start_str,
                sample_end=sample_end_str,
            )
    
            with st.status("运行尾盘策略…", expanded=True) as status:
                progress = st.progress(0.0, text="准备中…")
                try:
                    ranked = run_tail_strategy(cfg, status=status, progress=progress)
                    if ranked is None or ranked.empty:
                        status.update(label="微盘池为空或打分后无结果", state="error")
                    else:
                        status.update(label="策略运行完成", state="complete")
                        progress.progress(1.0, text="完成")
                        out_cols = ["代码", "名称", "score", "pct_chg_realtime",
                                    "tail_vol_ratio", "tail_slope", "流通市值", "换手率"]
                        out = ranked[[c for c in out_cols if c in ranked.columns]].head(cfg.top_n)
                        st.dataframe(out, use_container_width=True, hide_index=True)
                        st.download_button(
                            "导出 CSV",
                            out.to_csv(index=False).encode("utf-8-sig"),
                            file_name="tail_micro_result.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    status.update(label=f"运行失败：{e}", state="error")

# 20261120 双策略
elif page == "量比双策略":
    st.title("双策略")

    with st.expander("策略参数（可调上下限）", expanded=False):
        st.markdown("**量比触发区间（互斥）**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: vra_lo = st.number_input("A 量比下限", value=0.5, step=0.1)
        with c2: vra_hi = st.number_input("A 量比上限", value=1.2, step=0.1)
        with c3: vrb_lo = st.number_input("B 量比下限", value=1.5, step=0.1)
        with c4: vrb_hi = st.number_input("B 量比上限", value=3.0, step=0.1)

        st.markdown("**策略 A 条件**")
        a1, a2, a3 = st.columns(3)
        with a1:
            a_t_lo = st.number_input("换手下限 %", value=1.0, step=0.5)
            a_t_hi = st.number_input("换手上限 %", value=4.0, step=0.5)
        with a2:
            a_amt = st.number_input("成交额下限（万元）", value=5000.0, step=500.0)
            a_cap_lo = st.number_input("总市值下限（亿）", value=30.0, step=5.0)
        with a3:
            a_cap_hi = st.number_input("总市值上限（亿）", value=300.0, step=10.0)
            a_pct_lo = st.number_input("涨跌幅下限 %", value=-3.0, step=0.5)
            a_pct_hi = st.number_input("涨跌幅上限 %", value=3.0, step=0.5)

        st.markdown("**策略 B 条件**")
        b1, b2, b3 = st.columns(3)
        with b1:
            b_t_lo = st.number_input("换手下限 %", value=3.0, step=0.5)
            b_t_hi = st.number_input("换手上限 %", value=10.0, step=0.5)
        with b2:
            b_amt = st.number_input("成交额下限（万元）", value=20000.0, step=1000.0)
            b_cap_lo = st.number_input("总市值下限（亿）", value=50.0, step=10.0)
        with b3:
            b_cap_hi = st.number_input("总市值上限（亿）", value=500.0, step=50.0)
            b_pct_lo = st.number_input("涨跌幅下限 %", value=2.0, step=0.5)
            b_pct_hi = st.number_input("涨跌幅上限 %", value=5.0, step=0.5)

    run_dual = st.button("运行双策略筛选", type="primary")

    if run_dual:
        cfg = DualStrategyConfig(
            vol_ratio_a_low=vra_lo, vol_ratio_a_high=vra_hi,
            vol_ratio_b_low=vrb_lo, vol_ratio_b_high=vrb_hi,
            a_turnover_low=a_t_lo, a_turnover_high=a_t_hi,
            a_amount_min=a_amt * 1e4,
            a_mktcap_low=a_cap_lo * 1e8, a_mktcap_high=a_cap_hi * 1e8,
            a_pct_low=a_pct_lo, a_pct_high=a_pct_hi,
            b_turnover_low=b_t_lo, b_turnover_high=b_t_hi,
            b_amount_min=b_amt * 1e4,
            b_mktcap_low=b_cap_lo * 1e8, b_mktcap_high=b_cap_hi * 1e8,
            b_pct_low=b_pct_lo, b_pct_high=b_pct_hi,
        )
        with st.spinner("拉取实时快照并计算候选池均线（仅对初筛后的股票拉历史，约 10–30 秒）…"):
            result_a, result_b = run_dual_strategy(cfg)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader(f"策略 A 结果：{len(result_a)} 只")
            if result_a.empty:
                st.info("无符合条件的股票")
            else:
                st.dataframe(to_display(result_a), use_container_width=True, hide_index=True)
                st.download_button(
                    "导出 A 结果",
                    to_display(result_a).to_csv(index=False).encode("utf-8-sig"),
                    file_name="strategy_A.csv", mime="text/csv", key="dl_a",
                )
        with col_b:
            st.subheader(f"策略 B 结果：{len(result_b)} 只")
            if result_b.empty:
                st.info("无符合条件的股票")
            else:
                st.dataframe(to_display(result_b), use_container_width=True, hide_index=True)
                st.download_button(
                    "导出 B 结果",
                    to_display(result_b).to_csv(index=False).encode("utf-8-sig"),
                    file_name="strategy_B.csv", mime="text/csv", key="dl_b",
                )
