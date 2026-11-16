# -*- coding: utf-8 -*-
"""
Boyer-Moore 문자열 검색 시각화 FastAPI 백엔드 서버 (main.py)
- 나쁜 문자표(Bad Character Rule)와 착한 접미부(Good Suffix Rule)의 전처리 테이블 및
  단계별 히스토그램 스냅샷(Snapshot), 소요 시간, 시간 복잡도 진행률을 생성하여 반환합니다.
"""

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="AlgoCraft Boyer-Moore Visualizer API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_START_TIME = time.time()
SERVER_VERSION = "1.0.0"

ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    "bad_character": {
        "id": "bad_character",
        "name_ko": "보이어-무어 (나쁜 문자표 적용)",
        "name_en": "Bad Character Heuristic",
        "time_complexity": {
            "best": "O(N / M)",
            "average": "O(N / |Σ|)",
            "worst": "O(N × M)",
        },
        "space_complexity": "O(|Σ|)",
        "principle": (
            "패턴의 오른쪽 끝에서 왼쪽으로 비교하다 불일치 발생 시, 본문(Text)의 불일치 문자가 "
            "패턴 내 마지막으로 등장한 위치와 일치하도록 패턴을 우측으로 점프시킵니다."
        ),
    },
    "good_suffix": {
        "id": "good_suffix",
        "name_ko": "보이어-무어 (착한 접미부 적용)",
        "name_en": "Good Suffix Heuristic",
        "time_complexity": {
            "best": "O(N / M)",
            "average": "O(N)",
            "worst": "O(N + M)",
        },
        "space_complexity": "O(M)",
        "principle": (
            "불일치 발생 전까지 이미 일치했던 우측 접미부(Good Suffix)가 패턴 내부의 다른 위치에 "
            "존재하거나, 접미부의 부분 접미사가 패턴의 접두사와 일치하는 최대 거리만큼 점프합니다."
        ),
    },
    "combined_bm": {
        "id": "combined_bm",
        "name_ko": "보이어-무어 통합 (max(BC, GS))",
        "name_en": "Full Boyer-Moore (BC + GS)",
        "time_complexity": {
            "best": "O(N / M)",
            "average": "O(N / M)",
            "worst": "O(N + M)",
        },
        "space_complexity": "O(|Σ| + M)",
        "principle": (
            "매 불일치 시점마다 나쁜 문자 이동량(BC Shift)과 착한 접미부 이동량(GS Shift) 중 "
            "더 큰 이동 거리(max)를 선택하여 검색 횟수를 최소화합니다."
        ),
    },
}


class SearchRequest(BaseModel):
    text: str = Field(default="ABAAABCDABCDAABCABAAABCDAB", description="검색 대상 본문 문자열 (Text)")
    pattern: str = Field(default="ABCDA", description="찾고자 하는 패턴 문자열 (Pattern)")


def build_bad_character_table(pattern: str) -> Dict[str, int]:
    """패턴 내 각 문자의 마지막 등장 인덱스를 기록하는 나쁜 문자표(Bad Character Table)를 생성합니다."""
    table: Dict[str, int] = {}
    for idx, ch in enumerate(pattern):
        table[ch] = idx
    return table


def build_good_suffix_table(pattern: str) -> Dict[str, List[int]]:
    """패턴의 착한 접미부 이동 거리 배열(shift)과 경계 위치 배열(bpos)을 생성합니다."""
    m = len(pattern)
    shift = [0] * (m + 1)
    bpos = [0] * (m + 1)

    i = m
    j = m + 1
    bpos[i] = j

    # Case 1: 패턴 내부에 동일한 착한 접미부가 다시 등장하는 경우
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = bpos[j]
        i -= 1
        j -= 1
        bpos[i] = j

    # Case 2: 착한 접미부의 접미사가 패턴의 접두사(Prefix)와 일치하는 경우
    j = bpos[0]
    for k in range(m + 1):
        if shift[k] == 0:
            shift[k] = j
        if k == j:
            j = bpos[j]

    return {"shift": shift, "bpos": bpos}


def simulate_bm_mode(
    text: str,
    pattern: str,
    mode: str,
    bc_table: Dict[str, int],
    gs_shift: List[int],
) -> Dict[str, Any]:
    """
    지정된 전략(mode: 'bad_character' | 'good_suffix' | 'combined_bm')으로
    보이어-무어 검색을 수행하며 각 단계의 히스토그램 스냅샷 및 시간 복잡도 메타데이터를 생성합니다.
    """
    start_ns = time.perf_counter_ns()
    n = len(text)
    m = len(pattern)

    history: List[Dict[str, Any]] = []
    comp_freq = [0] * n
    found_indices: List[int] = []
    total_comparisons = 0
    total_shifts = 0

    theoretical_best = max(1.0, round(n / max(1, m), 2))
    theoretical_linear = n
    theoretical_worst = n * m

    unique_chars = sorted(set(text + pattern))
    char_rank = {ch: idx + 1 for idx, ch in enumerate(unique_chars)}
    max_rank = max(1, len(unique_chars))

    def record_step(
        phase: str,
        s_offset: int,
        p_idx: Optional[int],
        matched_window_indices: List[int],
        mismatch_text_idx: Optional[int],
        bc_step_shift: int,
        gs_step_shift: int,
        applied_shift: int,
        description: str,
    ) -> None:
        elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 4)
        progress_pct = min(
            100,
            max(0, round(((s_offset + m) / max(1, n)) * 100)),
        )

        # 각 본문 인덱스별 히스토그램 막대(Bar) 상태 스냅샷 구성
        histogram_bars: List[Dict[str, Any]] = []
        for idx in range(n):
            in_window = s_offset <= idx < s_offset + m
            pat_pos = idx - s_offset if in_window else None
            pat_char = pattern[pat_pos] if (pat_pos is not None and 0 <= pat_pos < m) else None

            if mismatch_text_idx == idx:
                bar_state = "mismatch"
            elif p_idx is not None and (s_offset + p_idx) == idx and phase == "compare":
                bar_state = "comparing"
            elif idx in matched_window_indices:
                bar_state = "matched"
            elif any(f_idx <= idx < f_idx + m for f_idx in found_indices):
                bar_state = "found"
            elif in_window:
                bar_state = "window"
            else:
                bar_state = "idle"

            # 히스토그램 높이(25~100%): 문자 코드 순위 + 누적 비교 횟수 가중치 반영
            base_h = int((char_rank.get(text[idx], 1) / max_rank) * 55) + 25
            freq_bonus = min(20, comp_freq[idx] * 6)
            bar_height_pct = min(100, base_h + freq_bonus)

            histogram_bars.append(
                {
                    "index": idx,
                    "text_char": text[idx],
                    "pattern_char": pat_char,
                    "height_pct": bar_height_pct,
                    "comparisons_here": comp_freq[idx],
                    "in_window": in_window,
                    "state": bar_state,
                }
            )

        complexity_ratio = round(total_comparisons / max(1, n), 3)
        history.append(
            {
                "step": len(history),
                "phase": phase,
                "shift_offset": s_offset,
                "pattern_index": p_idx,
                "text_index": (s_offset + p_idx) if p_idx is not None else None,
                "matched_indices": list(matched_window_indices),
                "mismatch_index": mismatch_text_idx,
                "found_occurrences": list(found_indices),
                "bc_shift": bc_step_shift,
                "gs_shift": gs_step_shift,
                "applied_shift": applied_shift,
                "histogram": histogram_bars,
                "metrics": {
                    "elapsed_ms": elapsed_ms,
                    "comparisons": total_comparisons,
                    "shifts": total_shifts,
                    "complexity_ratio_c_over_n": complexity_ratio,
                    "theoretical_best_ops": theoretical_best,
                    "theoretical_linear_ops": theoretical_linear,
                    "theoretical_worst_ops": theoretical_worst,
                    "progress_percent": progress_pct,
                },
                "description": description,
            }
        )

    # 초기 정렬 위치 스냅샷 (s = 0)
    record_step(
        phase="init",
        s_offset=0,
        p_idx=m - 1,
        matched_window_indices=[],
        mismatch_text_idx=None,
        bc_step_shift=0,
        gs_step_shift=0,
        applied_shift=0,
        description=f"[{ALGORITHM_METADATA[mode]['name_ko']}] 검색 시작 (본문 길이 N={n}, 패턴 길이 M={m}). 우측 끝 인덱스 j={m - 1}부터 역방향 비교합니다.",
    )

    s = 0
    while s <= n - m:
        j = m - 1
        matched_in_window: List[int] = []

        while j >= 0:
            t_idx = s + j
            total_comparisons += 1
            comp_freq[t_idx] += 1

            if pattern[j] == text[t_idx]:
                matched_in_window.append(t_idx)
                record_step(
                    phase="match",
                    s_offset=s,
                    p_idx=j,
                    matched_window_indices=matched_in_window,
                    mismatch_text_idx=None,
                    bc_step_shift=0,
                    gs_step_shift=0,
                    applied_shift=0,
                    description=f"[일치] 본문 T[{t_idx}]('{text[t_idx]}') == 패턴 P[{j}]('{pattern[j]}') 일치 (연속 일치 {len(matched_in_window)}글자)",
                )
                j -= 1
            else:
                bad_char = text[t_idx]
                last_occ = bc_table.get(bad_char, -1)
                bc_val = max(1, j - last_occ)
                gs_val = gs_shift[j + 1]

                if mode == "bad_character":
                    chosen_shift = bc_val
                    rule_desc = f"나쁜 문자표(BC): '{bad_char}' 기준 +{bc_val}칸 이동 (참고: GS는 +{gs_val}칸)"
                elif mode == "good_suffix":
                    chosen_shift = gs_val
                    rule_desc = f"착한 접미부(GS): 일치 접미부 '{pattern[j + 1 :]}' 기준 +{gs_val}칸 이동 (참고: BC는 +{bc_val}칸)"
                else:
                    chosen_shift = max(bc_val, gs_val)
                    rule_desc = f"통합(max): BC(+{bc_val}) vs GS(+{gs_val}) -> +{chosen_shift}칸 점프"

                total_shifts += 1
                record_step(
                    phase="mismatch_shift",
                    s_offset=s,
                    p_idx=j,
                    matched_window_indices=matched_in_window,
                    mismatch_text_idx=t_idx,
                    bc_step_shift=bc_val,
                    gs_step_shift=gs_val,
                    applied_shift=chosen_shift,
                    description=f"[불일치 & 점프] T[{t_idx}]('{bad_char}') != P[{j}]('{pattern[j]}') -> {rule_desc}",
                )
                s += chosen_shift
                break

        if j < 0:
            found_indices.append(s)
            bc_val = m - bc_table.get(text[s + m], -1) if (s + m < n) else 1
            gs_val = gs_shift[0]
            if mode == "bad_character":
                chosen_shift = max(1, bc_val)
            elif mode == "good_suffix":
                chosen_shift = max(1, gs_val)
            else:
                chosen_shift = max(bc_val, gs_val, 1)

            total_shifts += 1
            record_step(
                phase="found",
                s_offset=s,
                p_idx=0,
                matched_window_indices=matched_in_window,
                mismatch_text_idx=None,
                bc_step_shift=bc_val,
                gs_step_shift=gs_val,
                applied_shift=chosen_shift,
                description=f"🎉 [패턴 발견!] 본문 인덱스 [{s}..{s + m - 1}]에서 '{pattern}' 완전 일치! 다음 탐색을 위해 +{chosen_shift}칸 이동합니다.",
            )
            s += chosen_shift

    record_step(
        phase="done",
        s_offset=min(s, max(0, n - m)),
        p_idx=None,
        matched_window_indices=[],
        mismatch_text_idx=None,
        bc_step_shift=0,
        gs_step_shift=0,
        applied_shift=0,
        description=(
            f"[탐색 완료] 총 {len(history)} 스텝, 문자 비교 {total_comparisons}회, "
            f"패턴 이동 {total_shifts}회, 발견 위치: {found_indices if found_indices else '없음'}"
        ),
    )

    total_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 4)
    return {
        "mode": mode,
        "metadata": ALGORITHM_METADATA[mode],
        "total_steps": len(history),
        "total_comparisons": total_comparisons,
        "total_shifts": total_shifts,
        "execution_time_ms": total_ms,
        "found_occurrences": found_indices,
        "complexity_summary": {
            "text_length_n": n,
            "pattern_length_m": m,
            "comparisons_per_char": round(total_comparisons / max(1, n), 3),
            "best_case_ops": theoretical_best,
            "worst_case_ops": theoretical_worst,
        },
        "history": history,
    }


def run_boyer_moore_comparison(text: str, pattern: str) -> Dict[str, Any]:
    """나쁜 문자표(BC), 착한 접미부(GS), 통합(BC+GS) 3가지 모드를 실행하고 비교 데이터를 반환합니다."""
    overall_start = time.perf_counter_ns()
    bc_table = build_bad_character_table(pattern)
    gs_data = build_good_suffix_table(pattern)
    gs_shift = gs_data["shift"]

    bc_result = simulate_bm_mode(text, pattern, "bad_character", bc_table, gs_shift)
    gs_result = simulate_bm_mode(text, pattern, "good_suffix", bc_table, gs_shift)
    combined_result = simulate_bm_mode(text, pattern, "combined_bm", bc_table, gs_shift)

    max_timeline_steps = max(
        bc_result["total_steps"],
        gs_result["total_steps"],
        combined_result["total_steps"],
    )
    overall_ms = round((time.perf_counter_ns() - overall_start) / 1_000_000, 4)

    # 전처리 테이블 시각화용 데이터 구조화
    m = len(pattern)
    bc_table_view = [
        {"char": ch, "last_index": idx, "default_shift_from_end": (m - 1) - idx}
        for ch, idx in sorted(bc_table.items())
    ]
    gs_table_view = [
        {
            "mismatch_at_j": idx - 1 if idx > 0 else "완전일치(j=-1)",
            "matched_suffix": pattern[idx:] if idx < m else "(빈 접미부)",
            "gs_shift": gs_shift[idx],
            "border_pos": gs_data["bpos"][idx],
        }
        for idx in range(m + 1)
    ]

    return {
        "success": True,
        "server_status": "ONLINE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "pattern": pattern,
        "text_length": len(text),
        "pattern_length": len(pattern),
        "max_timeline_steps": max_timeline_steps,
        "total_server_execution_ms": overall_ms,
        "preprocessing_tables": {
            "bad_character_table": bc_table_view,
            "good_suffix_table": gs_table_view,
        },
        "algorithms": {
            "bad_character": bc_result,
            "good_suffix": gs_result,
            "combined_bm": combined_result,
        },
    }


@app.get("/health")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "success": True,
        "server_status": "ONLINE",
        "service": "algocraft-boyer-moore-fastapi",
        "version": SERVER_VERSION,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithms": ALGORITHM_METADATA,
    }


@app.post("/api/search")
@app.post("/api/bm-search")
def api_execute_bm_search(req: SearchRequest) -> JSONResponse:
    text = (req.text or "").strip()
    pattern = (req.pattern or "").strip()

    if len(text) < 2 or len(text) > 120:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "server_status": "ERROR",
                "error": {
                    "error_code": "ERR_TEXT_LENGTH",
                    "message": f"본문(Text) 길이는 2자 이상 120자 이하여야 합니다 (현재 {len(text)}자).",
                },
            },
        )
    if len(pattern) < 1 or len(pattern) > len(text):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "server_status": "ERROR",
                "error": {
                    "error_code": "ERR_PATTERN_LENGTH",
                    "message": "패턴(Pattern) 길이는 1자 이상이며 본문 길이 이하여야 합니다.",
                },
            },
        )

    result = run_boyer_moore_comparison(text, pattern)
    return JSONResponse(status_code=200, content=result)


@app.get("/api/search")
def api_get_bm_search(
    text: str = Query(default="ABAAABCDABCDAABCABAAABCDAB"),
    pattern: str = Query(default="ABCDA"),
) -> JSONResponse:
    return api_execute_bm_search(SearchRequest(text=text, pattern=pattern))


@app.get("/")
def serve_root() -> Any:
    gas_index = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gas", "index.html"))
    if os.path.exists(gas_index):
        return FileResponse(gas_index)
    return health_check()
