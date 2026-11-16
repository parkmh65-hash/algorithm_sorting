# Boyer-Moore Algorithm Comparison Server (FastAPI)

보이어-무어(Boyer-Moore) 문자열 검색 알고리즘 시각화를 위한 FastAPI 백엔드 서버입니다.
나쁜 문자 이동표(Bad Character Rule)와 착한 접미부 이동표(Good Suffix Rule)의 비교 및 단계별 막대그래프(히스토그램) 상태 스냅샷을 생성하여 반환합니다. Google Cloud Run 컨테이너 배포를 지원합니다.

## API 엔드포인트
- `GET /health`: 서버 헬스 체크
- `POST /api/search`: 보이어-무어 알고리즘 비교 실행 (텍스트, 패턴 입력)
- `POST /api/analyze`: 상세 분석 및 단계별 히스토그램 스냅샷 반환

## 배포 및 실행 방법

### 로컬 실행
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Google Cloud Run 배포
```bash
gcloud run deploy boyer-moore-server \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated
```
