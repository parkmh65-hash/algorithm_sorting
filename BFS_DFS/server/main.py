"""
================================================================================
[Google Cloud Run 서버] BFS vs DFS 미로 그래프 탐색 비교 API (FastAPI)
- 너비 우선 탐색 (BFS, Breadth-First Search, Queue FIFO)
- 깊이 우선 탐색 (DFS, Depth-First Search, Stack LIFO / Backtracking)
- 미로 생성 알고리즘 (Recursive Backtracker, Kruskal, Prim, Braid)
- 탐색 단계별 상태 스냅샷, 최단 경로 검증, 큐/스택 메모리 피크 및 복잡도 분석 반환
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
    version="2.4.0"
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
        "data_structure": "Queue (FIFO)",
        "time_complexity": {
            "best": "O(V + E)",
            "average": "O(V + E)",
            "worst": "O(V + E)"
        },
        "space_complexity": "O(V) (파면 프론티어 큐 메모리)",
        "optimality": "가중치 없는 그래프에서 최단 경로 100% 보장",
        "description": "출발점에서 가까운 노드부터 동심원 파면(Wavefront) 형태로 균등하게 전파하며 탐색합니다."
    },
    "dfs": {
        "name": "깊이 우선 탐색 (Depth-First Search)",
        "name_en": "Depth-First Search (DFS)",
        "data_structure": "Call Stack (LIFO) / 재귀 백트래킹",
        "time_complexity": {
            "best": "O(V + E)",
            "average": "O(V + E)",
            "worst": "O(V + E)"
        },
        "space_complexity": "O(h) (h: 최대 탐색 트리 깊이, 메모리 소비 극소화)",
        "optimality": "목표 노드를 먼저 발견할 뿐 최단 경로를 보장하지 않음",
        "description": "한 방향으로 막다른 길(Dead-end)을 만날 때까지 깊게 파고든 뒤 되돌아오는(Backtrack) 방식으로 탐색합니다."
    }
}


# ==============================================================================
# 2. Pydantic 요청/응답 모델
# ==============================================================================
class MazeGenerateRequest(BaseModel):
    size: int = Field(21, ge=9, le=61, description="미로 한 변의 크기 (홀수 권장, 9~61)")
    preset: str = Field("recursive_backtracker", description="미로 생성 알고리즘: recursive_backtracker, kruskal, prim, braid")
    seed: Optional[int] = Field(None, description="재현 가능한 난수 시드값")


class Coordinate(BaseModel):
    r: int
    c: int


class MazeSolveRequest(BaseModel):
    maze: List[List[int]] = Field(..., description="2D 미로 배열 (0: 벽, 1: 통로)")
    start: Optional[Coordinate] = Field(default=Coordinate(r=1, c=1), description="출발 지점 좌표")
    goal: Optional[Coordinate] = Field(default=None, description="도착 지점 좌표 (기본값: 우측 하단 통로)")


# ==============================================================================
# 3. 미로 생성 알고리즘 유틸리티
# ==============================================================================
def generate_maze(size: int, preset: str, seed: Optional[int] = None) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    if size % 2 == 0:
        size += 1  # 홀수 크기로 보정

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

    # 기본 생성 수행
    if preset in ["prim", "prim_dense"]:
        carve_prim()
    else:
        carve_backtracker()

    # Braid maze 변형: 막다른 길을 뚫어 루프(Cycle) 생성
    if preset in ["braid", "braid_labyrinth"]:
        for r in range(1, size - 1, 2):
            for c in range(1, size - 1, 2):
                if grid[r][c] == 1:
                    adj_count = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if grid[r + dr][c + dc] == 1)
                    if adj_count == 1:  # 막다른 길
                        # 벽 하나를 허물어 루프 형성
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
# 4. BFS & DFS 탐색 분석 엔진
# ==============================================================================
def solve_bfs(maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    queue = deque([start])
    visited = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    visited_order: List[Tuple[int, int]] = []
    queue_depth_history: List[int] = []
    peak_queue_size = 1
    ops = 0
    found = False

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        ops += 1
        curr = queue.popleft()
        visited_order.append(curr)
        queue_depth_history.append(len(queue))

        if curr == goal:
            found = True
            break

        cr, cc = curr
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = curr
                    queue.append((nr, nc))
                    if len(queue) > peak_queue_size:
                        peak_queue_size = len(queue)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    # 경로 역추적
    path: List[Tuple[int, int]] = []
    if found:
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

    return {
        "algorithm": "BFS",
        "found": found,
        "path": [{"r": r, "c": c} for r, c in path],
        "path_length": len(path),
        "visited_count": len(visited),
        "visited_nodes": [{"r": r, "c": c} for r, c in visited_order],
        "peak_memory_units": peak_queue_size,
        "memory_history": queue_depth_history[:100],  # 상위 100개 샘플
        "operations": ops,
        "elapsed_ms": elapsed_ms,
        "optimality_guaranteed": True
    }


def solve_dfs(maze: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    stack = [start]
    visited = {start}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    visited_order: List[Tuple[int, int]] = []
    stack_depth_history: List[int] = []
    peak_stack_size = 1
    ops = 0
    found = False

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while stack:
        ops += 1
        curr = stack.pop()
        visited_order.append(curr)
        stack_depth_history.append(len(stack))

        if curr == goal:
            found = True
            break

        cr, cc = curr
        # DFS는 스택 순서(역순 삽입)에 맞춰 탐색
        for dr, dc in reversed(dirs):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = curr
                    stack.append((nr, nc))
                    if len(stack) > peak_stack_size:
                        peak_stack_size = len(stack)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    # 경로 역추적
    path: List[Tuple[int, int]] = []
    if found:
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()

    return {
        "algorithm": "DFS",
        "found": found,
        "path": [{"r": r, "c": c} for r, c in path],
        "path_length": len(path),
        "visited_count": len(visited),
        "visited_nodes": [{"r": r, "c": c} for r, c in visited_order],
        "peak_memory_units": peak_stack_size,
        "memory_history": stack_depth_history[:100],  # 상위 100개 샘플
        "operations": ops,
        "elapsed_ms": elapsed_ms,
        "optimality_guaranteed": False
    }


# ==============================================================================
# 5. API 엔드포인트 (/health, /api/generate, /api/solve)
# ==============================================================================
@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "ALGO:VISION - BFS vs DFS Maze Graph Explorer API",
        "version": "2.4.0",
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

    bfs_res = solve_bfs(maze, start_pt, goal_pt)
    dfs_res = solve_dfs(maze, start_pt, goal_pt)

    # 비교 분석 메트릭 산출
    path_delta = dfs_res["path_length"] - bfs_res["path_length"]
    visited_delta = bfs_res["visited_count"] - dfs_res["visited_count"]

    comparison = {
        "path_length_delta": path_delta,
        "path_comparison": "동일 최단 경로" if path_delta == 0 else f"DFS가 BFS보다 +{path_delta} 스텝 긴 비최적 경로",
        "visited_nodes_delta": visited_delta,
        "visited_comparison": (
            f"DFS가 BFS보다 {visited_delta}개 적은 노드 탐색" if visited_delta > 0
            else f"BFS가 DFS보다 {-visited_delta}개 적은 노드 탐색"
        ),
        "memory_peak_ratio": round(bfs_res["peak_memory_units"] / max(1, dfs_res["peak_memory_units"]), 2),
        "bfs_optimal": True,
        "dfs_optimal": path_delta == 0
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
