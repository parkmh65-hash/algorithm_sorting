# Fashion AI Comparative Synthesizer & Multi-modal RAG API

FastAPI 기반의 멀티모달 패션 스타일링 RAG 추천 백엔드 서비스입니다. Google Cloud Run 컨테이너 배포를 지원합니다.

## 배포 및 실행 방법

### 로컬 실행
```bash
pip install -r requirements.txt
python main.py
```

### Google Cloud Run 배포
```bash
gcloud run deploy fashion-ai-backend \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated
```