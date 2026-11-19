"""
================================================================================
[Google Cloud Run 서버] BFS vs DFS 미로 그래프 탐색 비교 API (FastAPI)
- 너비 우선 탐색 (BFS, Breadth-First Search, Queue FIFO)
- 깊이 우선 탐색 (DFS, Depth-First Search, Stack LIFO / Backtracking)
- 미로 생성 알고리즘 (Recursive Backtracker, Kruskal, Prim, Braid)
- 단계별 히스토그램 스냅샷 데이터 생성 및 시간 복잡도 진행 상태 메타데이터 반환
================================================================================
"""

import time
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="ALGO:VISION - BFS vs DFS Maze Graph Explorer API",
    version="2.5.0",
    description="BFS vs DFS 미로 탐출 알고리즘 단계별 히스토그램 및 복잡도 분석 엔진"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_private_network_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


# ==============================================================================
# 1. 알고리즘 복잡도 및 메타데이터 정의
# ==============================================================================
ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    "bfs": {
        "name": "너비 우선 탐색 (Breadth-First Search)",
        "name_en": "Breadth-First Search (BFS)",
        "data_structure": "Queue (FIFO - First In First Out)",
        "time_complexity": {
            "theoretical": "O(V + E)",
            "description": "모든 정점(V)과 간선(E)을 레벨(파면) 순서로 1회씩 방문",
            "progress_formula": "Visited Nodes / Total Passable Nodes"
        },
        "space_complexity": {
            "theoretical": "O(V)",
            "description": "최대 파면(Frontier) 폭에 비례하여 큐 메모리 급증 (최악 시 O(V))"
        },
        "optimality": "가중치 없는 그래프에서 최단 경로 100% 보장",
        "search_pattern": "동심원 파면(Wavefront) 균등 확장 탐색",
        "key_operation": "Enqueue (후방 추가) & Dequeue (전방 추출)"
    },
    "dfs": {
        "name": "깊이 우선 탐색 (Depth-First Search)",
        "name_en": "Depth-First Search (DFS)",
        "data_structure": "Call Stack (LIFO - Last In First Out) / 재귀 백트래킹",
        "time_complexity": {
            "theoretical": "O(V + E)",
            "description": "단일 경로를 끝까지 탐색 후 역추적(Backtracking)하여 방문",
            "progress_formula": "Visited Nodes / Total Passable Nodes"
        },
        "space_complexity": {
            "theoretical": "O(h) (h = 트리 최대 깊이)",
            "description": "현재 분기 경로 깊이에만 비례하여 메모리 소비 극소화"
        },
        "optimality": "목표 노드를 먼저 발견할 뿐 최단 경로를 보장하지 않음 (비최적 경로 가능)",
        "search_pattern": "단일 회랑(Corridor) 심층 직진 및 막다른 길 후퇴",
        "key_operation": "Push (스택 탑 추가) & Pop (스택 탑 추출/백트래킹)"
    }
}


# ==============================================================================
# 2. Pydantic 요청/응답 모델
# ==============================================================================
class MazeGenerateRequest(BaseModel):
    size: int = Field(21, ge=9, le=61, description="미로 한 변의 크기 (홀수 권장, 9~61)")
    preset: str = Field("recursive_backtracker", description="미로 생성 알고리즘: recursive_backtracker, prim, braid")
    seed: Optional[int] = Field(None, description="재현 가능한 난수 시드값")


class Coordinate(BaseModel):
    r: int
    c: int


class MazeSolveRequest(BaseModel):
    maze: List[List[int]] = Field(..., description="2D 미로 배열 (0: 벽, 1: 통로)")
    start: Optional[Coordinate] = Field(default=Coordinate(r=1, c=1), description="출발 지점 좌표")
    goal: Optional[Coordinate] = Field(default=None, description="도착 지점 좌표 (기본값: 우측 하단 통로)")
    max_steps: Optional[int] = Field(default=1000, description="스냅샷 최대 기록 단계 수")


# ==============================================================================
# 3. 미로 생성 알고리즘 유틸리티
# ==============================================================================
def generate_maze(size: int, preset: str, seed: Optional[int] = None) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    if size % 2 == 0:
        size += 1

    if seed is not None:
        random.seed(seed)

    grid = [[0 for _ in range(size)] for _ in range(size)]

    # 1) Recursive Backtracker
    def carve_backtracker():
        stack = [(1, 1)]
        grid[1][1] = 1
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            cr, cc = stack[-1]
            neighbors = []
            for dr, dc in dirs:
                nr, nc = cr + dr, cc + dc
                if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr][nc] == 0:
                    neighbors.append((dr, dc, nr, nc))

            if neighbors:
                dr, dc, nr, nc = random.choice(neighbors)
                grid[cr + dr // 2][cc + dc // 2] = 1
                grid[nr][nc] = 1
                stack.append((nr, nc))
            else:
                stack.pop()

    # 2) Prim's Algorithm
    def carve_prim():
        grid[1][1] = 1
        walls = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = 1 + dr, 1 + dc
            if 0 < nr < size - 1 and 0 < nc < size - 1:
                walls.append((1, 1, nr, nc, dr, dc))

        while walls:
            idx = random.randint(0, len(walls) - 1)
            fr, fc, to_r, to_c, dr, dc = walls.pop(idx)

            if grid[to_r][to_c] == 0:
                grid[fr + dr // 2][fc + dc // 2] = 1
                grid[to_r][to_c] = 1

                for next_dr, next_dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    nr, nc = to_r + next_dr, to_c + next_dc
                    if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr][nc] == 0:
                        walls.append((to_r, to_c, nr, nc, next_dr, next_dc))

    if preset in ["prim", "prim_dense"]:
        carve_prim()
    else:
        carve_backtracker()

    # Braid maze 변형: 막다른 길을 뚫어 다중 경로 생성
    if preset in ["braid", "braid_labyrinth"]:
        for r in range(1, size - 1, 2):
            for c in range(1, size - 1, 2):
                if grid[r][c] == 1:
                    adj_count = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if grid[r + dr][c + dc] == 1)
                    if adj_count == 1:
                        candidates = []
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr][nc] == 0:
                                candidates.append((nr, nc))
                        if candidates:
                            wr, wc = random.choice(candidates)
                            grid[wr][wc] = 1

    start = {"r": 1, "c": 1}
    goal = {"r": size - 2, "c": size - 2}
    grid[start["r"]][start["c"]] = 1
    grid[goal["r"]][goal["c"]] = 1

    return grid, start, goal


# ==============================================================================
# 4. BFS 단계별 히스토그램 & 스냅샷 생성기
# ==============================================================================
def solve_bfs_with_snapshots(
    maze: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 1000
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    total_passable = sum(row.count(1) for row in maze)

    queue = deque([(start[0], start[1], 0)])  # (r, c, depth)
    visited = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    visited_order: List[Dict[str, int]] = []
    snapshots: List[Dict[str, Any]] = []
    histogram_history: List[Dict[str, Any]] = []

    peak_queue_size = 1
    step = 0
    found = False
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 동, 남, 서, 북

    # 초기 스냅샷
    histogram_history.append({
        "step": 0,
        "queue_size": 1,
        "visited_count": 1,
        "depth": 0,
        "complexity_ratio": round(1 / max(1, total_passable), 4),
        "elapsed_ms": 0.0
    })

    while queue:
        step += 1
        step_t0 = time.perf_counter()
        cr, cc, depth = queue.popleft()
        visited_order.append({"r": cr, "c": cc})

        if (cr, cc) == goal:
            found = True
            step_elapsed = round((time.perf_counter() - step_t0) * 1000, 3)
            cum_elapsed = round((time.perf_counter() - t_start) * 1000, 3)

            snap = {
                "step": step,
                "current_node": {"r": cr, "c": cc},
                "queue_size": len(queue),
                "visited_count": len(visited),
                "depth": depth,
                "action": "goal_reached",
                "step_elapsed_ms": step_elapsed,
                "cumulative_ms": cum_elapsed,
                "complexity_ratio": round(len(visited) / max(1, total_passable), 4)
            }
            if len(snapshots) < max_steps:
                snapshots.append(snap)
            histogram_history.append({
                "step": step,
                "queue_size": len(queue),
                "visited_count": len(visited),
                "depth": depth,
                "complexity_ratio": snap["complexity_ratio"],
                "elapsed_ms": cum_elapsed
            })
            break

        enqueued_nodes = []
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (cr, cc)
                    queue.append((nr, nc, depth + 1))
                    enqueued_nodes.append({"r": nr, "c": nc})
                    if len(queue) > peak_queue_size:
                        peak_queue_size = len(queue)

        step_elapsed = round((time.perf_counter() - step_t0) * 1000, 3)
        cum_elapsed = round((time.perf_counter() - t_start) * 1000, 3)

        snap = {
            "step": step,
            "current_node": {"r": cr, "c": cc},
            "queue_size": len(queue),
            "visited_count": len(visited),
            "depth": depth,
            "enqueued_count": len(enqueued_nodes),
            "action": "dequeue_and_expand",
            "step_elapsed_ms": step_elapsed,
            "cumulative_ms": cum_elapsed,
            "complexity_ratio": round(len(visited) / max(1, total_passable), 4)
        }
        if len(snapshots) < max_steps:
            snapshots.append(snap)

        histogram_history.append({
            "step": step,
            "queue_size": len(queue),
            "visited_count": len(visited),
            "depth": depth,
            "complexity_ratio": snap["complexity_ratio"],
            "elapsed_ms": cum_elapsed
        })

    total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 3)

    # 최단 경로 역추적
    path: List[Dict[str, int]] = []
    if found:
        curr = goal
        while curr is not None:
            path.append({"r": curr[0], "c": curr[1]})
            curr = parent[curr]
        path.reverse()

    return {
        "algorithm": "BFS",
        "data_structure": "Queue (FIFO)",
        "found": found,
        "path": path,
        "path_length": len(path),
        "total_visited": len(visited),
        "visited_nodes": visited_order,
        "peak_queue_size": peak_queue_size,
        "total_operations": step,
        "total_elapsed_ms": total_elapsed_ms,
        "avg_step_ms": round(total_elapsed_ms / max(1, step), 4),
        "optimality_guaranteed": True,
        "snapshots": snapshots[:200],  # UI 렌더링용 상위 스냅샷
        "histogram_data": histogram_history[:100],  # 단계별 히스토그램 데이터
        "metadata": ALGORITHM_METADATA["bfs"]
    }


# ==============================================================================
# 5. DFS 단계별 히스토그램 & 스냅샷 생성기
# ==============================================================================
def solve_dfs_with_snapshots(
    maze: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    max_steps: int = 1000
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    total_passable = sum(row.count(1) for row in maze)

    # stack: [(r, c, path_from_start)]
    stack: List[Tuple[int, int, List[Tuple[int, int]]]] = [(start[0], start[1], [start])]
    visited = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    visited_order: List[Dict[str, int]] = []
    snapshots: List[Dict[str, Any]] = []
    histogram_history: List[Dict[str, Any]] = []

    peak_stack_size = 1
    step = 0
    found = False
    final_path: List[Dict[str, int]] = []
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 역순 탐색

    histogram_history.append({
        "step": 0,
        "stack_size": 1,
        "visited_count": 1,
        "depth": 1,
        "complexity_ratio": round(1 / max(1, total_passable), 4),
        "elapsed_ms": 0.0
    })

    while stack:
        step += 1
        step_t0 = time.perf_counter()
        cr, cc, curr_path = stack[-1]  # Top
        visited_order.append({"r": cr, "c": cc})

        if (cr, cc) == goal:
            found = True
            final_path = [{"r": r, "c": c} for r, c in curr_path]
            step_elapsed = round((time.perf_counter() - step_t0) * 1000, 3)
            cum_elapsed = round((time.perf_counter() - t_start) * 1000, 3)

            snap = {
                "step": step,
                "current_node": {"r": cr, "c": cc},
                "stack_size": len(stack),
                "visited_count": len(visited),
                "depth": len(stack),
                "action": "goal_reached",
                "step_elapsed_ms": step_elapsed,
                "cumulative_ms": cum_elapsed,
                "complexity_ratio": round(len(visited) / max(1, total_passable), 4)
            }
            if len(snapshots) < max_steps:
                snapshots.append(snap)
            histogram_history.append({
                "step": step,
                "stack_size": len(stack),
                "visited_count": len(visited),
                "depth": len(stack),
                "complexity_ratio": snap["complexity_ratio"],
                "elapsed_ms": cum_elapsed
            })
            break

        # 미방문 이웃 탐색 (단일 방향 깊이 전진)
        found_next = False
        for dr, dc in reversed(dirs):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (cr, cc)
                    stack.append((nr, nc, curr_path + [(nr, nc)]))
                    if len(stack) > peak_stack_size:
                        peak_stack_size = len(stack)
                    found_next = True
                    break

        action = "stack_push"
        if not found_next:
            # 막다른 길: 백트래킹 (스택 팝)
            stack.pop()
            action = "stack_pop_backtrack"

        step_elapsed = round((time.perf_counter() - step_t0) * 1000, 3)
        cum_elapsed = round((time.perf_counter() - t_start) * 1000, 3)

        snap = {
            "step": step,
            "current_node": {"r": cr, "c": cc},
            "stack_size": len(stack),
            "visited_count": len(visited),
            "depth": len(stack),
            "action": action,
            "step_elapsed_ms": step_elapsed,
            "cumulative_ms": cum_elapsed,
            "complexity_ratio": round(len(visited) / max(1, total_passable), 4)
        }
        if len(snapshots) < max_steps:
            snapshots.append(snap)

        histogram_history.append({
            "step": step,
            "stack_size": len(stack),
            "visited_count": len(visited),
            "depth": len(stack),
            "complexity_ratio": snap["complexity_ratio"],
            "elapsed_ms": cum_elapsed
        })

    total_elapsed_ms = round((time.perf_counter() - t_start) * 1000, 3)

    return {
        "algorithm": "DFS",
        "data_structure": "Call Stack (LIFO)",
        "found": found,
        "path": final_path,
        "path_length": len(final_path),
        "total_visited": len(visited),
        "visited_nodes": visited_order,
        "peak_stack_size": peak_stack_size,
        "total_operations": step,
        "total_elapsed_ms": total_elapsed_ms,
        "avg_step_ms": round(total_elapsed_ms / max(1, step), 4),
        "optimality_guaranteed": False,
        "snapshots": snapshots[:200],  # UI 렌더링용 상위 스냅샷
        "histogram_data": histogram_history[:100],  # 단계별 히스토그램 데이터
        "metadata": ALGORITHM_METADATA["dfs"]
    }


# ==============================================================================
# 6. API 엔드포인트 (/health, /api/generate, /api/solve)
# ==============================================================================
@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ALGO:VISION - BFS vs DFS Maze Graph Explorer API",
        "version": "2.5.0",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "supported_algorithms": ["BFS", "DFS"],
        "supported_maze_presets": [
            "recursive_backtracker",
            "prim",
            "braid"
        ],
        "metadata": ALGORITHM_METADATA
    }


@app.post("/api/generate")
def generate_maze_endpoint(req: MazeGenerateRequest) -> JSONResponse:
    grid, start, goal = generate_maze(size=req.size, preset=req.preset, seed=req.seed)
    return JSONResponse(content={
        "status": "success",
        "preset": req.preset,
        "size": len(grid),
        "start": start,
        "goal": goal,
        "maze": grid,
        "cell_count": len(grid) * len(grid[0]),
        "passable_cells": sum(row.count(1) for row in grid),
        "timestamp": time.time()
    })


@app.post("/api/solve")
def solve_maze_endpoint(req: MazeSolveRequest) -> JSONResponse:
    maze = req.maze
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    if rows < 5 or cols < 5:
        return JSONResponse(status_code=400, content={"status": "error", "message": "미로 크기는 최소 5x5 이상이어야 합니다."})

    start_pt = (req.start.r, req.start.c)
    goal_pt = (req.goal.r, req.goal.c) if req.goal else (rows - 2, cols - 2)

    bfs_res = solve_bfs_with_snapshots(maze, start_pt, goal_pt, max_steps=req.max_steps or 1000)
    dfs_res = solve_dfs_with_snapshots(maze, start_pt, goal_pt, max_steps=req.max_steps or 1000)

    # 비교 분석 메트릭 산출
    path_delta = dfs_res["path_length"] - bfs_res["path_length"]
    visited_delta = bfs_res["total_visited"] - dfs_res["total_visited"]

    comparison = {
        "path_length_delta": path_delta,
        "path_comparison": "동일 최단 경로" if path_delta == 0 else f"DFS가 BFS보다 +{path_delta} 스텝 긴 비최적 경로",
        "visited_nodes_delta": visited_delta,
        "visited_comparison": (
            f"DFS가 BFS보다 {visited_delta}개 적은 노드 탐색" if visited_delta > 0
            else f"BFS가 DFS보다 {-visited_delta}개 적은 노드 탐색"
        ),
        "memory_peak_ratio": round(bfs_res["peak_queue_size"] / max(1, dfs_res["peak_stack_size"]), 2),
        "bfs_optimal": True,
        "dfs_optimal": path_delta == 0,
        "time_complexity_summary": {
            "bfs": "O(V + E) - 완전 파면 전파",
            "dfs": "O(V + E) - 단일 분기 백트래킹"
        },
        "space_complexity_summary": {
            "bfs": f"O(V) (Peak Queue: {bfs_res['peak_queue_size']} 노드)",
            "dfs": f"O(h) (Peak Stack: {dfs_res['peak_stack_size']} 노드)"
        }
    }

    return JSONResponse(content={
        "status": "success",
        "start": {"r": start_pt[0], "c": start_pt[1]},
        "goal": {"r": goal_pt[0], "c": goal_pt[1]},
        "grid_dimensions": {"rows": rows, "cols": cols},
        "bfs": bfs_res,
        "dfs": dfs_res,
        "comparison": comparison,
        "metadata": ALGORITHM_METADATA
    })
