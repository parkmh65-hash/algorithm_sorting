"""
================================================================================
[Google Cloud Run 서버] 정렬 알고리즘 상태 스냅샷(State Snapshot) 생성 API 서버
================================================================================
이 모듈은 셸 정렬(Shell Sort), 퀵 정렬(Quick Sort), 병합 정렬(Merge Sort) 3가지 알고리즘을
끝까지 수행하면서, 단순히 최종 정렬 결과만 반환하는 것이 아니라 각 알고리즘의 고유한 동작 원리가
드러나는 매 순간의 '상태 스냅샷(State Snapshot)'을 배열에 기록하여 클라이언트로 전달합니다.

주요 기능:
1. GET  /health     : 서버 동작 상태 및 버전, 지원 알고리즘 정보를 반환 (상태 점검용)
2. POST /api/sort   : 입력 배열을 받아 3가지 정렬 알고리즘의 전체 상태 히스토리와 복잡도 정보를 반환
================================================================================
"""

import os
import time
import traceback
from typing import Any, Dict, List, Set
from flask import Flask, jsonify, request
from flask_cors import CORS

# Flask 애플리케이션 초기화 및 CORS(교차 출처 리소스 공유) 허용 설정
# GAS(UrlFetchApp)뿐만 아니라 로컬 브라우저에서도 직접 호출할 수 있도록 허용합니다.
app = Flask(__name__)
CORS(app)

# 입력 배열 검증 상수 설정
MIN_ARRAY_SIZE = 2
MAX_ARRAY_SIZE = 60
MIN_VALUE = -9999
MAX_VALUE = 9999


# ==============================================================================
# 1. 알고리즘 복잡도 및 메타데이터 정의
# ==============================================================================
ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    "shell_sort": {
        "name": "셸 정렬 (Shell Sort)",
        "name_en": "Shell Sort",
        "time_complexity": {
            "best": "O(n log n)",
            "average": "O(n^1.25) ~ O(n^1.5)",
            "worst": "O(n²)"
        },
        "space_complexity": "O(1) (제자리 정렬, In-place)",
        "stable": False,
        "principle": (
            "일정한 간격(Gap)만큼 떨어진 요소들끼리 부분 그룹을 묶어 삽입 정렬을 수행하고, "
            "간격을 점진적으로 절반씩 줄여가며(Gap = N/2 → N/4 → ... → 1) 전체 배열의 정렬 상태를 "
            "빠르게 개선하는 알고리즘입니다."
        )
    },
    "quick_sort": {
        "name": "퀵 정렬 (Quick Sort)",
        "name_en": "Quick Sort",
        "time_complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n²) (이미 정렬된 배열에서 양끝 피벗 선택 시)"
        },
        "space_complexity": "O(log n) (재귀 호출 스택)",
        "stable": False,
        "principle": (
            "분할 정복(Divide and Conquer) 기반으로 기준점인 피벗(Pivot)을 지정한 뒤, "
            "피벗보다 작은 요소는 왼쪽, 큰 요소는 오른쪽으로 분할(Partition)하고 피벗의 최종 위치를 "
            "확정해 나가는 고속 정렬 알고리즘입니다."
        )
    },
    "merge_sort": {
        "name": "병합 정렬 (Merge Sort)",
        "name_en": "Merge Sort",
        "time_complexity": {
            "best": "O(n log n)",
            "average": "O(n log n)",
            "worst": "O(n log n) (항상 일정한 성능 보장)"
        },
        "space_complexity": "O(n) (병합용 임시 배열 사용)",
        "stable": True,
        "principle": (
            "배열을 크기가 1이 될 때까지 논리적으로 반씩 분할(Divide)한 다음, "
            "두 부분 배열의 좌우 요소를 순서대로 비교하여 임시 배열에 정렬한 뒤 원본 배열에 "
            "다시 덮어씌우는(Overwrite) 안정 정렬 알고리즘입니다."
        )
    }
}


# ==============================================================================
# 2. 셸 정렬 (Shell Sort) 스냅샷 생성 함수
# ==============================================================================
def generate_shell_sort_snapshots(initial_array: List[int]) -> Dict[str, Any]:
    """
    셸 정렬을 수행하며 다음 핵심 순간들을 스냅샷으로 기록합니다:
    - 간격(Gap)의 크기 변화
    - 설정된 간격만큼 떨어진 요소들 간의 그룹화(gap_group) 및 비교(comparing)
    - 요소가 삽입/교환(swapping)되어 이동하는 상태
    - 최종 정렬 완료 상태(sorted_indices)
    """
    arr = list(initial_array)
    n = len(arr)
    snapshots: List[Dict[str, Any]] = []
    comparisons = 0
    writes = 0

    def record(
        phase: str,
        description: str,
        gap: int,
        gap_group: List[int] = None,
        comparing: List[int] = None,
        swapping: List[int] = None,
        sorted_indices: List[int] = None,
        active_range: List[int] = None
    ) -> None:
        """현재 셸 정렬의 상태를 하나의 스냅샷 딕셔너리로 히스토리에 추가합니다."""
        snapshots.append({
            "step": len(snapshots),
            "algorithm": "shell_sort",
            "phase": phase,
            "array": list(arr),
            "gap": gap,
            "gap_group": gap_group or [],
            "comparing": comparing or [],
            "swapping": swapping or [],
            "sorted_indices": sorted_indices or [],
            "active_range": active_range or [0, n - 1],
            "stats": {
                "comparisons": comparisons,
                "writes": writes
            },
            "description": description
        })

    # 초기 상태 기록
    gap = n // 2
    record(
        phase="INIT",
        description=f"셸 정렬을 시작합니다. 초기 배열 크기 N={n}, 첫 번째 간격(Gap)={gap}으로 설정합니다.",
        gap=gap
    )

    # 간격(Gap)을 절반씩 줄여가며 1이 될 때까지 반복
    while gap > 0:
        record(
            phase="GAP_CHANGE",
            description=f"간격(Gap)을 {gap}(으)로 설정하여 {gap}칸씩 떨어진 요소들을 그룹화합니다.",
            gap=gap
        )

        # 현재 gap 간격의 부분 배열들에 대해 삽입 정렬 수행
        for i in range(gap, n):
            temp = arr[i]
            j = i

            # 현재 요소 i와 동일한 Gap 그룹에 속한 인덱스들을 계산하여 시각화에 표시
            group_offset = i % gap
            current_gap_group = list(range(group_offset, i + 1, gap))

            record(
                phase="GROUP_SELECT",
                description=(
                    f"[Gap={gap}] 인덱스 {i}(값: {temp})를 선택하고, "
                    f"동일 간격 그룹 {current_gap_group} 내에서 삽입 위치를 탐색합니다."
                ),
                gap=gap,
                gap_group=current_gap_group,
                comparing=[j - gap, j]
            )

            # Gap만큼 앞선 요소와 비교하며 삽입 위치 탐색
            while j >= gap:
                comparisons += 1
                record(
                    phase="COMPARE",
                    description=(
                        f"[Gap={gap}] 간격만큼 떨어진 인덱스 {j - gap}(값: {arr[j - gap]})와 "
                        f"삽입 대상 값({temp})을 비교합니다."
                    ),
                    gap=gap,
                    gap_group=current_gap_group,
                    comparing=[j - gap, j]
                )

                if arr[j - gap] > temp:
                    # 앞선 요소가 더 크므로 Gap만큼 오른쪽으로 이동(교환 효과)
                    arr[j] = arr[j - gap]
                    arr[j - gap] = temp  # 시각적 직관성을 위해 이동 즉시 temp를 반영
                    writes += 1

                    record(
                        phase="SWAP",
                        description=(
                            f"[Gap={gap}] {arr[j]} > {temp}이므로 두 요소를 교환(이동)합니다: "
                            f"인덱스 {j - gap} ↔ 인덱스 {j}"
                        ),
                        gap=gap,
                        gap_group=current_gap_group,
                        swapping=[j - gap, j]
                    )
                    j -= gap
                else:
                    # 올바른 위치를 찾았으므로 비교 종료
                    record(
                        phase="KEEP",
                        description=(
                            f"[Gap={gap}] {arr[j - gap]} ≤ {temp}이므로 현재 위치(인덱스 {j})에 "
                            f"값 {temp}의 배치를 확정합니다."
                        ),
                        gap=gap,
                        gap_group=current_gap_group,
                        comparing=[j - gap, j]
                    )
                    break

            arr[j] = temp

        # 다음 단계를 위해 Gap을 절반으로 축소
        gap //= 2

    # 최종 정렬 완료 스냅샷 기록
    record(
        phase="COMPLETE",
        description=f"셸 정렬이 완료되었습니다! (총 비교: {comparisons}회, 이동/교환: {writes}회)",
        gap=0,
        sorted_indices=list(range(n))
    )

    return {
        "snapshots": snapshots,
        "total_steps": len(snapshots),
        "comparisons": comparisons,
        "writes": writes,
        "sorted_array": arr
    }


# ==============================================================================
# 3. 퀵 정렬 (Quick Sort) 스냅샷 생성 함수
# ==============================================================================
def generate_quick_sort_snapshots(initial_array: List[int]) -> Dict[str, Any]:
    """
    퀵 정렬을 수행하며 다음 핵심 순간들을 스냅샷으로 기록합니다:
    - 피벗(Pivot) 지정 및 현재 분할 구간(low ~ high)
    - 피벗과의 대소 비교 대상(comparing)
    - 요소 교환 위치(swapping)
    - 피벗의 최종 정렬 확정 상태(sorted_indices)
    """
    arr = list(initial_array)
    n = len(arr)
    snapshots: List[Dict[str, Any]] = []
    sorted_indices: Set[int] = set()
    comparisons = 0
    writes = 0

    def record(
        phase: str,
        description: str,
        pivot_index: int = None,
        comparing: List[int] = None,
        swapping: List[int] = None,
        partition_range: List[int] = None
    ) -> None:
        """현재 퀵 정렬의 상태를 하나의 스냅샷 딕셔너리로 히스토리에 추가합니다."""
        snapshots.append({
            "step": len(snapshots),
            "algorithm": "quick_sort",
            "phase": phase,
            "array": list(arr),
            "pivot_index": pivot_index,
            "pivot_value": arr[pivot_index] if pivot_index is not None else None,
            "comparing": comparing or [],
            "swapping": swapping or [],
            "active_range": partition_range or [0, n - 1],
            "sorted_indices": sorted(list(sorted_indices)),
            "stats": {
                "comparisons": comparisons,
                "writes": writes
            },
            "description": description
        })

    # 초기 상태 기록
    record(
        phase="INIT",
        description=f"퀵 정렬을 시작합니다. 전체 구간 [0..{n - 1}]에 대해 분할 정복을 수행합니다.",
        partition_range=[0, n - 1]
    )

    def quick_sort_recursive(low: int, high: int) -> None:
        nonlocal comparisons, writes

        # 원소가 1개 이하인 구간은 그 자체로 정렬 확정
        if low >= high:
            if 0 <= low < n:
                sorted_indices.add(low)
                record(
                    phase="SINGLE_CONFIRMED",
                    description=f"단일 원소 구간 [인덱스 {low}](값: {arr[low]})의 최종 위치를 확정합니다.",
                    partition_range=[low, low]
                )
            return

        # 로무토 분할(Lomuto Partition) 방식: 구간의 마지막 요소를 피벗(Pivot)으로 지정
        pivot_idx = high
        pivot_val = arr[pivot_idx]
        i = low - 1  # 피벗보다 작거나 같은 요소들이 쌓일 경계 인덱스

        record(
            phase="PIVOT_SELECT",
            description=(
                f"구간 [{low}..{high}]에서 인덱스 {pivot_idx}(값: {pivot_val})를 "
                f"피벗(Pivot)으로 지정합니다."
            ),
            pivot_index=pivot_idx,
            partition_range=[low, high]
        )

        for j in range(low, high):
            comparisons += 1
            record(
                phase="COMPARE",
                description=(
                    f"인덱스 {j}(값: {arr[j]})와 피벗(인덱스 {pivot_idx}, 값: {pivot_val})의 "
                    f"대소를 비교합니다."
                ),
                pivot_index=pivot_idx,
                comparing=[j, pivot_idx],
                partition_range=[low, high]
            )

            if arr[j] <= pivot_val:
                i += 1
                if i != j:
                    arr[i], arr[j] = arr[j], arr[i]
                    writes += 1
                    record(
                        phase="SWAP",
                        description=(
                            f"{arr[i]} ≤ 피벗({pivot_val})이므로 왼쪽 구역 확장을 위해 "
                            f"인덱스 {i}(값: {arr[i]})와 인덱스 {j}(값: {arr[j]})를 교환합니다."
                        ),
                        pivot_index=pivot_idx,
                        swapping=[i, j],
                        partition_range=[low, high]
                    )
                else:
                    record(
                        phase="KEEP",
                        description=(
                            f"{arr[j]} ≤ 피벗({pivot_val})이며 이미 왼쪽 경계(인덱스 {i})에 "
                            f"위치하므로 자리를 유지합니다."
                        ),
                        pivot_index=pivot_idx,
                        comparing=[i],
                        partition_range=[low, high]
                    )

        # 피벗을 최종 분할 경계(i + 1) 위치로 이동
        final_pivot_idx = i + 1
        if final_pivot_idx != high:
            arr[final_pivot_idx], arr[high] = arr[high], arr[final_pivot_idx]
            writes += 1
            record(
                phase="PIVOT_SWAP",
                description=(
                    f"피벗({pivot_val})을 분할 경계 위치인 인덱스 {final_pivot_idx}(으)로 "
                    f"교환하여 이동합니다."
                ),
                pivot_index=final_pivot_idx,
                swapping=[final_pivot_idx, high],
                partition_range=[low, high]
            )

        # 피벗의 최종 정렬 위치 확정 기록
        sorted_indices.add(final_pivot_idx)
        record(
            phase="PIVOT_CONFIRMED",
            description=(
                f"피벗 값 {pivot_val}의 최종 정렬 위치가 인덱스 {final_pivot_idx}에 확정되었습니다!"
            ),
            pivot_index=final_pivot_idx,
            partition_range=[low, high]
        )

        # 피벗 기준 왼쪽 부분 배열과 오른쪽 부분 배열에 대해 재귀 수행
        quick_sort_recursive(low, final_pivot_idx - 1)
        quick_sort_recursive(final_pivot_idx + 1, high)

    quick_sort_recursive(0, n - 1)

    # 최종 완료 스냅샷
    sorted_indices = set(range(n))
    record(
        phase="COMPLETE",
        description=f"퀵 정렬이 완료되었습니다! (총 비교: {comparisons}회, 교환: {writes}회)",
        partition_range=[0, n - 1]
    )

    return {
        "snapshots": snapshots,
        "total_steps": len(snapshots),
        "comparisons": comparisons,
        "writes": writes,
        "sorted_array": arr
    }


# ==============================================================================
# 4. 병합 정렬 (Merge Sort) 스냅샷 생성 함수
# ==============================================================================
def generate_merge_sort_snapshots(initial_array: List[int]) -> Dict[str, Any]:
    """
    병합 정렬을 수행하며 다음 핵심 순간들을 스냅샷으로 기록합니다:
    - 배열이 논리적으로 분할되는 구간(left ~ mid ~ right)
    - 두 부분 배열을 병합하며 좌우 요소를 비교하는 과정(comparing)
    - 임시 배열(temp_buffer)에서 정렬된 값이 원본 배열로 덮어씌워지는(overwriting) 상태
    """
    arr = list(initial_array)
    n = len(arr)
    snapshots: List[Dict[str, Any]] = []
    comparisons = 0
    writes = 0

    def record(
        phase: str,
        description: str,
        left: int = 0,
        mid: int = None,
        right: int = None,
        comparing: List[int] = None,
        overwriting: List[int] = None,
        temp_buffer: List[int] = None,
        sorted_indices: List[int] = None
    ) -> None:
        """현재 병합 정렬의 상태를 하나의 스냅샷 딕셔너리로 히스토리에 추가합니다."""
        r = right if right is not None else n - 1
        snapshots.append({
            "step": len(snapshots),
            "algorithm": "merge_sort",
            "phase": phase,
            "array": list(arr),
            "active_range": [left, r],
            "mid": mid,
            "comparing": comparing or [],
            "overwriting": overwriting or [],
            "temp_buffer": list(temp_buffer) if temp_buffer is not None else [],
            "sorted_indices": sorted_indices or [],
            "stats": {
                "comparisons": comparisons,
                "writes": writes
            },
            "description": description
        })

    # 초기 상태 기록
    record(
        phase="INIT",
        description=f"병합 정렬을 시작합니다. 전체 배열 [0..{n - 1}]을 반씩 분할합니다.",
        left=0,
        right=n - 1
    )

    def merge_sort_recursive(left: int, right: int) -> None:
        nonlocal comparisons, writes

        if left >= right:
            return

        mid = (left + right) // 2

        # 1단계: 논리적 분할 구간 기록
        record(
            phase="DIVIDE",
            description=(
                f"구간 [{left}..{right}]을 왼쪽 [{left}..{mid}]와 "
                f"오른쪽 [{mid + 1}..{right}] 두 부분 배열로 분할합니다."
            ),
            left=left,
            mid=mid,
            right=right
        )

        # 좌우 부분 배열 재귀 분할
        merge_sort_recursive(left, mid)
        merge_sort_recursive(mid + 1, right)

        # 2단계: 두 부분 배열을 비교하며 임시 배열(temp_buffer)에 병합
        temp_buffer: List[int] = []
        i = left
        j = mid + 1

        record(
            phase="MERGE_START",
            description=(
                f"정렬된 왼쪽 구간 [{left}..{mid}]와 오른쪽 구간 [{mid + 1}..{right}]의 "
                f"병합(Merge)을 시작합니다."
            ),
            left=left,
            mid=mid,
            right=right
        )

        while i <= mid and j <= right:
            comparisons += 1
            record(
                phase="COMPARE",
                description=(
                    f"왼쪽 요소 인덱스 {i}(값: {arr[i]})와 "
                    f"오른쪽 요소 인덱스 {j}(값: {arr[j]})를 비교합니다."
                ),
                left=left,
                mid=mid,
                right=right,
                comparing=[i, j],
                temp_buffer=temp_buffer
            )

            if arr[i] <= arr[j]:
                temp_buffer.append(arr[i])
                record(
                    phase="TEMP_COLLECT",
                    description=(
                        f"왼쪽 값({arr[i]}) ≤ 오른쪽 값({arr[j]})이므로 "
                        f"임시 배열에 {arr[i]}을(를) 추가합니다. (임시 배열: {temp_buffer})"
                    ),
                    left=left,
                    mid=mid,
                    right=right,
                    comparing=[i],
                    temp_buffer=temp_buffer
                )
                i += 1
            else:
                temp_buffer.append(arr[j])
                record(
                    phase="TEMP_COLLECT",
                    description=(
                        f"오른쪽 값({arr[j]}) < 왼쪽 값({arr[i]})이므로 "
                        f"임시 배열에 {arr[j]}을(를) 추가합니다. (임시 배열: {temp_buffer})"
                    ),
                    left=left,
                    mid=mid,
                    right=right,
                    comparing=[j],
                    temp_buffer=temp_buffer
                )
                j += 1

        # 왼쪽 부분 배열에 남은 요소들을 임시 배열에 추가
        while i <= mid:
            temp_buffer.append(arr[i])
            record(
                phase="TEMP_REMAINDER",
                description=(
                    f"왼쪽 구간에 남은 요소 인덱스 {i}(값: {arr[i]})를 "
                    f"임시 배열에 추가합니다. (임시 배열: {temp_buffer})"
                ),
                left=left,
                mid=mid,
                right=right,
                comparing=[i],
                temp_buffer=temp_buffer
            )
            i += 1

        # 오른쪽 부분 배열에 남은 요소들을 임시 배열에 추가
        while j <= right:
            temp_buffer.append(arr[j])
            record(
                phase="TEMP_REMAINDER",
                description=(
                    f"오른쪽 구간에 남은 요소 인덱스 {j}(값: {arr[j]})를 "
                    f"임시 배열에 추가합니다. (임시 배열: {temp_buffer})"
                ),
                left=left,
                mid=mid,
                right=right,
                comparing=[j],
                temp_buffer=temp_buffer
            )
            j += 1

        # 3단계: 임시 배열(temp_buffer)에서 정렬 완료된 값들을 원본 배열(arr)로 덮어쓰기(Overwrite)
        for offset, val in enumerate(temp_buffer):
            target_idx = left + offset
            arr[target_idx] = val
            writes += 1

            # 전체 구간 병합인 경우 덮어쓴 위치까지 정렬 확정 표시
            is_final_merge = (left == 0 and right == n - 1)
            confirmed = list(range(0, target_idx + 1)) if is_final_merge else []

            record(
                phase="OVERWRITE",
                description=(
                    f"[Overwrite] 임시 배열의 {offset}번째 값({val})을 "
                    f"원본 배열 인덱스 {target_idx}에 덮어씌웁니다."
                ),
                left=left,
                mid=mid,
                right=right,
                overwriting=[target_idx],
                temp_buffer=temp_buffer,
                sorted_indices=confirmed
            )

    merge_sort_recursive(0, n - 1)

    # 최종 완료 스냅샷
    record(
        phase="COMPLETE",
        description=f"병합 정렬이 완료되었습니다! (총 비교: {comparisons}회, 덮어쓰기: {writes}회)",
        left=0,
        right=n - 1,
        sorted_indices=list(range(n))
    )

    return {
        "snapshots": snapshots,
        "total_steps": len(snapshots),
        "comparisons": comparisons,
        "writes": writes,
        "sorted_array": arr
    }


# ==============================================================================
# 5. 입력 데이터 검증 유틸리티 함수
# ==============================================================================
def validate_input_array(payload: Any) -> List[int]:
    """
    클라이언트가 전달한 JSON 요청 본문에서 'array' 필드를 추출하고 엄격하게 검증합니다.
    검증 실패 시 구체적인 원인을 담은 ValueError를 발생시킵니다.
    """
    if payload is None or not isinstance(payload, dict):
        raise ValueError(
            "[INVALID_JSON_BODY] 요청 본문(JSON Body)이 비어 있거나 올바른 JSON 객체 형식이 아닙니다."
        )

    if "array" not in payload:
        raise ValueError(
            "[MISSING_ARRAY_FIELD] 요청 JSON에 필수 필드인 'array'가 누락되었습니다."
        )

    raw_array = payload["array"]

    # 문자열 형태로 전달된 경우(예: "8, 3, 5, 1") 쉼표 분리 처리 지원
    if isinstance(raw_array, str):
        parts = [p.strip() for p in raw_array.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError("[EMPTY_INPUT_STRING] 입력된 문자열에 숫자가 포함되어 있지 않습니다.")
        try:
            raw_array = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"[NON_INTEGER_TOKEN] 정수(Integer)로 변환할 수 없는 값이 포함되어 있습니다: {exc}"
            ) from exc

    if not isinstance(raw_array, list):
        raise ValueError(
            f"[INVALID_ARRAY_TYPE] 'array' 필드는 배열(List) 또는 쉼표 구분 문자열이어야 합니다. "
            f"(전달된 타입: {type(raw_array).__name__})"
        )

    if len(raw_array) < MIN_ARRAY_SIZE:
        raise ValueError(
            f"[ARRAY_TOO_SMALL] 정렬을 비교하려면 최소 {MIN_ARRAY_SIZE}개 이상의 숫자가 필요합니다. "
            f"(현재 개수: {len(raw_array)}개)"
        )

    if len(raw_array) > MAX_ARRAY_SIZE:
        raise ValueError(
            f"[ARRAY_TOO_LARGE] 시각화 성능 보호를 위해 최대 {MAX_ARRAY_SIZE}개까지만 입력할 수 있습니다. "
            f"(현재 개수: {len(raw_array)}개)"
        )

    validated: List[int] = []
    for idx, item in enumerate(raw_array):
        if isinstance(item, bool):
            raise ValueError(
                f"[BOOLEAN_NOT_ALLOWED] {idx}번째 요소({item})가 불리언(Boolean)입니다. 정수를 입력해 주세요."
            )
        try:
            num = int(item)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"[INVALID_ELEMENT_VALUE] {idx}번째 요소('{item}')를 정수로 변환할 수 없습니다."
            ) from exc

        if num < MIN_VALUE or num > MAX_VALUE:
            raise ValueError(
                f"[VALUE_OUT_OF_RANGE] {idx}번째 요소({num})가 허용 범위({MIN_VALUE} ~ {MAX_VALUE})를 벗어났습니다."
            )
        validated.append(num)

    return validated


# ==============================================================================
# 6. HTTP API 엔드포인트 라우팅
# ==============================================================================
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """
    서버의 정상 동작 여부와 지원 알고리즘의 복잡도 메타데이터를 반환하는 상태 확인 엔드포인트입니다.
    """
    return jsonify({
        "status": "ok",
        "service": "Algorithm Visualizer Snapshot Server (Google Cloud Run)",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supported_algorithms": ["shell_sort", "quick_sort", "merge_sort"],
        "limits": {
            "min_array_size": MIN_ARRAY_SIZE,
            "max_array_size": MAX_ARRAY_SIZE,
            "min_value": MIN_VALUE,
            "max_value": MAX_VALUE
        },
        "metadata": ALGORITHM_METADATA
    }), 200


@app.route("/api/sort", methods=["POST"])
def execute_all_sorts():
    """
    클라이언트로부터 원본 배열을 전달받아 셸 정렬, 퀵 정렬, 병합 정렬을 각각 끝까지 실행하고
    매 순간의 상태 스냅샷(State Snapshot) 히스토리와 통합 타임라인 정보를 반환합니다.
    """
    start_time = time.perf_counter()

    try:
        # JSON 파싱 확인
        payload = request.get_json(silent=True)
        input_array = validate_input_array(payload)

        # 3가지 정렬 알고리즘 실행 및 상태 스냅샷 기록
        shell_result = generate_shell_sort_snapshots(input_array)
        quick_result = generate_quick_sort_snapshots(input_array)
        merge_result = generate_merge_sort_snapshots(input_array)

        # 단일 통합 타임라인 재생을 위한 최대 스텝 수 계산
        max_steps = max(
            shell_result["total_steps"],
            quick_result["total_steps"],
            merge_result["total_steps"]
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return jsonify({
            "status": "ok",
            "server_info": {
                "service": "Algorithm Visualizer Snapshot Server",
                "execution_time_ms": elapsed_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "input_array": input_array,
            "sorted_array": merge_result["sorted_array"],
            "timeline": {
                "max_steps": max_steps,
                "recommended_interval_ms": 500
            },
            "metadata": ALGORITHM_METADATA,
            "algorithms": {
                "shell_sort": shell_result,
                "quick_sort": quick_result,
                "merge_sort": merge_result
            }
        }), 200

    except ValueError as val_err:
        # 입력값 검증 오류 (400 Bad Request)
        return jsonify({
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "http_status": 400,
                "message": str(val_err),
                "hint": (
                    f"{MIN_ARRAY_SIZE}~{MAX_ARRAY_SIZE}개의 정수 배열을 전달해 주세요. "
                    "예: {\"array\": [38, 27, 43, 3, 9, 82, 10]}"
                )
            }
        }), 400

    except Exception as unexpected_err:
        # 예기치 못한 서버 내부 오류 (500 Internal Server Error)
        return jsonify({
            "status": "error",
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "http_status": 500,
                "message": f"서버 내부에서 정렬 스냅샷 생성 중 오류가 발생했습니다: {str(unexpected_err)}",
                "trace": traceback.format_exc()
            }
        }), 500


# ==============================================================================
# 7. 서버 엔트리포인트 (로컬 실행 및 Cloud Run 컨테이너 실행 겸용)
# ==============================================================================
if __name__ == "__main__":
    # Google Cloud Run은 환경변수 PORT(기본 8080)를 통해 수신 포트를 지정합니다.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

