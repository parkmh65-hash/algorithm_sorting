# ALGO:VISION - BFS vs DFS Maze Graph Explorer Server (FastAPI)

미로 그래프 상에서 너비 우선 탐색(BFS)과 깊이 우선 탐색(DFS) 알고리즘의 동작 원리 및 상태 스냅샷을 실시간으로 생성하여 비교하는 FastAPI 백엔드 서버입니다.
Google Cloud Run 컨테이너 배포 및 Google Apps Script 웹앱 연동을 지원합니다.

## 주요 기능
- **BFS (Breadth-First Search)**: 큐(FIFO) 기반 파면(Wavefront) 전파 및 가중치 없는 미로에서 최단 경로 100% 보장
- **DFS (Depth-First Search)**: 콜 스택(LIFO) 기반 한 우물 파기 탐색 및 막다른 길에서의 역추적(Backtracking)
- **미로 생성 엔진**: Recursive Backtracker, Kruskal, Prim, Braid Labyrinth 알고리즘 탑재
- **실시간 계측 및 복잡도 분석**: 경로 길이 차이(Delta), 탐색 노드 수, 큐/스택 피크 메모리 점유율, 시간/공간 복잡도 비교

## API 엔드포인트
- `GET /health`: 서버 헬스 체크 및 알고리즘 메타데이터 반환
- `POST /api/generate`: 지정된 크기 및 프리셋 미로 매트릭스 생성
- `POST /api/solve`: 입력 미로에 대해 BFS 및 DFS 동시 탐색 및 비교 메트릭 반환

## 로컬 실행 방법
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
