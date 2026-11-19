# Fashion Multi-modal RAG & VLM Server (FastAPI)

Fashionpedia 기반 멀티모달 RAG 및 Gemini VLM 스타일링 추천 백엔드 서버입니다.
Google Cloud Run 컨테이너 배포 및 GAS(Google Apps Script) 연동을 지원합니다.

## 주요 기능
- **이미지 청킹 (Image Chunking)**: 상의, 하의, 슈즈/액세서리 3대 영역 분할
- **Joint Embedding**: CLIP 스타일 시각 특징 및 텍스트 벡터 임베딩 (512-D)
- **ChromaDB Multi-Vector Retriever**: 다중 자식 벡터 검색 및 부모 Docstore 연동
- **Google AI Gemini VLM**: 멀티모달 스타일링 분석 및 한/영 번역 파이프라인

## API 엔드포인트
- `GET /health`: 서버 상태 점검 및 인덱싱 현황 조회
- `POST /query`: 사용자 패션 질의 및 이미지 멀티모달 RAG 스타일링 추천 (NDJSON 스트리밍 지원)
- `POST /build_db`: 신규 이미지 청킹 및 ChromaDB 인덱싱

## 실행 방법
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```
