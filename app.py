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

# 已登录，以下才是业务代码
st.sidebar.success(f"已登录：{st.session_state['name']}")
authenticator.logout(location="sidebar")

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


# ========== 6. 界面主流程 ==========
st.title("选股筛选器")
st.sidebar.caption(f"当前用户：{current_name}")
authenticator.logout(location="sidebar")

# 1) 先渲染筛选控件，这一步不加载数据
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
                if loaded:
                    apply_strategy_to_state(loaded)
                    st.session_state["loaded_strategy_name"] = selected
                    st.rerun()
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
    
# 2) 按钮行
col_run, col_refresh = st.columns([1, 1])
with col_run:
    run = st.button("开始筛选", type="primary")
with col_refresh:
    refresh = st.button("强制刷新数据")

# 3) 手动刷新：清缓存 + 清上次结果
if refresh:
    load_data.clear()
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_total", None)
    st.info("缓存已清除，请点击「开始筛选」重新拉取数据。")

# 8.4 点击才加载数据 + 筛选
if run:
    with st.spinner("正在加载行情数据并筛选，请稍候..."):
        df = add_pre_close(load_data())
        result = filter_stocks(df, filters)
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
                "ltsz", "turnover", "zljlr", "pre_close",
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
