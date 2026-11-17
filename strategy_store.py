# strategy_store.py
import base64
import json
from datetime import datetime, timezone

import requests
import streamlit as st

API = "https://api.github.com"


def _cfg():
    return st.secrets["github"]


def _headers():
    return {
        "Authorization": f"Bearer {_cfg()['token']}",
        "Accept": "application/vnd.github+json",
    }


def _url():
    return f"{API}/repos/{_cfg()['repo']}/contents/{_cfg()['path']}"


@st.cache_data(ttl=30)
def _read_raw() -> tuple[dict, str | None]:
    """返回 (全部策略, 当前文件 sha)。文件不存在时返回 ({}, None)。"""
    r = requests.get(_url(), headers=_headers(), timeout=10)
    if r.status_code == 404:
        return {}, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return json.loads(content), data["sha"]


def _write_raw(all_data: dict, sha: str | None, message: str):
    payload = {
        "message": message,
        "content": base64.b64encode(
            json.dumps(all_data, ensure_ascii=False, indent=2).encode("utf-8")
        ).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(_url(), headers=_headers(), json=payload, timeout=10)
    if r.status_code == 409:
        raise RuntimeError("策略文件已被其他人更新，请刷新后重试")
    r.raise_for_status()


def list_strategies(username: str) -> list[str]:
    all_data, _ = _read_raw()
    return sorted(all_data.get(username, {}).keys())


def load_strategy(username: str, name: str) -> dict | None:
    all_data, _ = _read_raw()
    item = all_data.get(username, {}).get(name)
    return item["filters"] if item else None


def save_strategy(username: str, name: str, filters: dict):
    all_data, sha = _read_raw()
    all_data.setdefault(username, {})[name] = {
        "filters": filters,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_raw(all_data, sha, f"strategy: save {username}/{name}")
    _read_raw.clear()  # 写后清缓存，下次读拿到最新


def delete_strategy(username: str, name: str):
    all_data, sha = _read_raw()
    if username in all_data and name in all_data[username]:
        del all_data[username][name]
        if not all_data[username]:
            del all_data[username]
        _write_raw(all_data, sha, f"strategy: delete {username}/{name}")
        _read_raw.clear()
