# ==============================================================================
# Google Cloud Run 배포용 Dockerfile
# ==============================================================================
# 가벼운 공식 Python 3.12 slim 이미지를 베이스로 사용합니다.
FROM python:3.12-slim

# 컨테이너 내부에서 파이썬 출력 버퍼링을 비활성화하여 Cloud Run 로그에 즉시 출력되도록 설정
# .pyc 바이트코드 파일 생성을 방지하여 컨테이너 용량을 최적화합니다.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 패키지 설치 (Docker 레이어 캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서버 소스 코드 전체 복사
COPY . .

# Google Cloud Run 기본 수신 포트 설정 (기본값: 8080)
ENV PORT=8080
EXPOSE 8080

# 프로덕션 WSGI 서버(Gunicorn)를 사용하여 애플리케이션 실행
# 워커 1개, 스레드 8개, 타임아웃 0(Cloud Run 자체 타임아웃에 위임)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

