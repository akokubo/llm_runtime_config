# -*- coding: utf-8 -*-
"""
ローカル LLM 設定ユーティリティ

要件:
- プラットフォーム: macOS / WSL（自動判定 or 明示指定）
- バックエンド: Ollama / LM Studio
- 共通パラメータ: api_key, temperature, sig_tau, sig_topk
- base_url:
    - Ollama:  http://localhost:11434/v1
    - LM Studio (macOS): http://localhost:1234/v1
    - LM Studio (WSL):   http://{get_windows_host_ip()}:1234/v1
- モデル名はバックエンド毎にエイリアスが異なるため、正規化（canonical_name -> backend別）で吸収

使い方例:
    python llm_runtime_config.py --backend lmstudio --platform auto \
        --model gemma3-4b-it-qat --api_key ollama

LangChain の ChatOpenAI を作る場合:
    from langchain_openai import ChatOpenAI
    cfg = resolve_config()
    llm = cfg.make_langchain_llm()

"""
from __future__ import annotations

import argparse
import os
import platform as py_platform
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

# ===============================
# ユーティリティ
# ===============================

def get_windows_host_ip() -> str:
    """WSL から Windows 側のホスト IP を推定（LM Studio 既定: 1234 番）"""
    # 1) resolv.conf の nameserver を最優先
    try:
        with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        if ip.count(".") == 3:
                            return ip
    except Exception:
        pass

    # 2) ルートテーブルのデフォルトゲートウェイ
    try:
        # awk のクォートに注意（単引用符で括る）
        cmd = "ip route show default | awk '{print $3}'"
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        ip = out.decode().strip().split()[0]
        if ip and ip.count(".") == 3:
            return ip
    except Exception:
        pass

    # 3) フォールバック
    return "127.0.0.1"


def is_wsl() -> bool:
    try:
        # /proc/version に Microsoft が含まれるか、uname リリースに microsoft が含まれるか
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            ver = f.read().lower()
            if "microsoft" in ver:
                return True
    except Exception:
        pass
    try:
        rel = os.uname().release.lower()
        if "microsoft" in rel:
            return True
    except Exception:
        pass
    return False


def detect_platform() -> str:
    if is_wsl():
        return "wsl"
    system = py_platform.system().lower()
    if system == "darwin":
        return "macos"
    # ここでは "linux" (非 WSL) や "windows" の場合もあるが、要件にないため macOS にフォールバック
    return "macos"

# ===============================
# モデル名エイリアス
# ===============================
# 正規化名 -> バックエンド毎モデル名
# 必要に応じて追加してください
MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    # Gemma 3 4B IT QAT の例
    "gemma3-4b-it-qat": {
        "ollama": "gemma3:4b-it-qat",     # 例: Ollama のタグ表記
        "lmstudio": "gemma3-4b-it-qat",   # 例: LM Studio 側のモデル名
    },
    # 例: Qwen2.5 7B Instruct
    "qwen2.5-7b-instruct": {
        "ollama": "qwen2.5:7b-instruct",
        "lmstudio": "qwen2.5-7b-instruct",
    },
}


def resolve_model_name(canonical: str, backend: str, fallback: Optional[str] = None) -> str:
    be = backend.lower()
    if canonical in MODEL_ALIASES and be in MODEL_ALIASES[canonical]:
        return MODEL_ALIASES[canonical][be]
    # 未登録のときは、そのまま返すかフォールバック
    return fallback or canonical

# ===============================
# 設定データクラス
# ===============================

@dataclass
class LLMConfig:
    backend: str                 # "ollama" | "lmstudio"
    platform: str                # "macos" | "wsl"
    model_canonical: str         # 例: "gemma3-4b-it-qat"

    # 共通パラメータ
    api_key: str                 # ローカルの場合は任意の文字列でOK（例: "ollama"）
    temperature: float = 0.7
    sig_tau: float = 0.7
    sig_topk: int = 5

    # 自動 or 明示の base_url
    base_url: Optional[str] = None

    # 明示的に backend 固有モデル名を指定したい場合（通常は自動解決）
    model_override: Optional[str] = None

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        be = self.backend.lower()
        pf = self.platform.lower()
        if be == "ollama":
            return "http://localhost:11434/v1"
        if be == "lmstudio":
            if pf == "macos":
                return "http://localhost:1234/v1"
            if pf == "wsl":
                return f"http://{get_windows_host_ip()}:1234/v1"
        # デフォルト（念のため）
        return "http://localhost:1234/v1"

    def resolved_model(self) -> str:
        if self.model_override:
            return self.model_override
        return resolve_model_name(self.model_canonical, self.backend)

    # LangChain 用のラッパ（任意）
    def make_langchain_llm(self):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "langchain_openai が見つかりません。`pip install langchain-openai` を実行してください"
            ) from e

        return ChatOpenAI(
            model=self.resolved_model(),
            base_url=self.resolved_base_url(),
            api_key=self.api_key,
            temperature=self.temperature,
        )

    # OpenAI 互換 SDK を直接使う場合のペイロード例
    def to_openai_kwargs(self) -> dict:
        return {
            "model": self.resolved_model(),
            "base_url": self.resolved_base_url(),
            "api_key": self.api_key,
            # 必要に応じて temperature などをメッセージレベルで渡す
        }


# ===============================
# 引数・環境変数
# ===============================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ローカル LLM 設定（Ollama / LM Studio）")
    p.add_argument("--backend", choices=["ollama", "lmstudio"], default=os.getenv("LLM_BACKEND", "lmstudio"))
    p.add_argument("--platform", choices=["auto", "macos", "wsl"], default=os.getenv("LLM_PLATFORM", "auto"))
    p.add_argument("--model", dest="model_canonical", default=os.getenv("LLM_MODEL", "gemma3-4b-it-qat"))

    # 共通パラメータ
    p.add_argument("--api_key", default=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "ollama")))
    p.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.7")))
    p.add_argument("--sig_tau", type=float, default=float(os.getenv("LLM_SIG_TAU", "0.7")))
    p.add_argument("--sig_topk", type=int, default=int(os.getenv("LLM_SIG_TOPK", "5")))

    # 上書き系
    p.add_argument("--base_url", default=os.getenv("LLM_BASE_URL"))
    p.add_argument("--model_override", default=os.getenv("LLM_MODEL_OVERRIDE"))
    return p


def resolve_platform_arg(arg: str) -> str:
    if arg == "auto":
        return detect_platform()
    return arg


def resolve_config(args: Optional[argparse.Namespace] = None) -> LLMConfig:
    if args is None:
        args = build_arg_parser().parse_args()

    cfg = LLMConfig(
        backend=args.backend,
        platform=resolve_platform_arg(args.platform),
        model_canonical=args.model_canonical,
        api_key=args.api_key,
        temperature=args.temperature,
        sig_tau=args.sig_tau,
        sig_topk=args.sig_topk,
        base_url=args.base_url,
        model_override=args.model_override,
    )
    return cfg


# ===============================
# スクリプト実行
# ===============================
if __name__ == "__main__":
    cfg = resolve_config()

    print("[Resolved Config]")
    print(f"  backend      : {cfg.backend}")
    print(f"  platform     : {cfg.platform}")
    print(f"  base_url     : {cfg.resolved_base_url()}")
    print(f"  model        : {cfg.resolved_model()}")
    print(f"  api_key      : {cfg.api_key}")
    print(f"  temperature  : {cfg.temperature}")
    print(f"  sig_tau      : {cfg.sig_tau}")
    print(f"  sig_topk     : {cfg.sig_topk}")

    # 動作テスト（任意）: LangChain が入っていれば 1 トーク投げる
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        llm = cfg.make_langchain_llm()
        resp = llm.invoke("日本語で一言、自己紹介してください。")
        print("\n[LLM Response]\n", getattr(resp, "content", resp))
    except Exception as e:
        print("\n[Note] LangChain でのテストはスキップしました:", e)
