/**
 * ============================================================================
 * [Google Apps Script 백엔드 브릿지] code.gs
 * 파일 경로: BFS_DFS/gas/code.gs
 * 역할:
 *   1. Google Cloud Run에 배포된 FastAPI(server/main.py) 서버와 UrlFetchApp으로 통신
 *   2. Cloud Run 서버 URL을 스크립트 환경변수(PropertiesService)에 안전하게 저장/조회
 *   3. GitHub Actions 배포 후 전달되는 doPost(e) 웹훅을 통해 Cloud Run URL 자동 갱신
 *   4. BFS vs DFS 미로 생성 및 탐색 분석 API 호출 중계
 * ============================================================================
 */

const ENV_CLOUD_RUN_URL_KEY = 'CLOUD_RUN_URL';
const DEFAULT_CLOUD_RUN_URL = 'http://localhost:8080';

/**
 * 웹앱 메인 HTML 화면 서빙
 */
function doGet(e) {
  return HtmlService.createTemplateFromFile('index')
    .evaluate()
    .setTitle('ALGO:VISION / BFS vs DFS Maze Graph Explorer')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL)
    .addMetaTag('viewport', 'width=device-width, initial-scale=1');
}

/**
 * Cloud Run 서비스 URL 조회
 */
function getCloudRunUrl() {
  const props = PropertiesService.getScriptProperties();
  let url = props.getProperty(ENV_CLOUD_RUN_URL_KEY);
  if (!url || url.trim() === '') {
    url = DEFAULT_CLOUD_RUN_URL;
  }
  return url.trim().replace(/\/+$/, '');
}

/**
 * Cloud Run 서비스 URL 저장
 */
function saveCloudRunUrl(newUrl) {
  if (!newUrl || typeof newUrl !== 'string' || !newUrl.trim().startsWith('http')) {
    return {
      ok: false,
      error: '유효한 http:// 또는 https:// 형식의 Cloud Run URL을 입력해 주세요.'
    };
  }
  const cleanUrl = newUrl.trim().replace(/\/+$/, '');
  PropertiesService.getScriptProperties().setProperty(ENV_CLOUD_RUN_URL_KEY, cleanUrl);
  return {
    ok: true,
    url: cleanUrl,
    message: 'Cloud Run URL이 스크립트 속성에 안전하게 저장되었습니다.'
  };
}

/**
 * [gas-url-sync 스킬 연동] GitHub Actions 배포 완료 웹훅 수신 핸들러
 */
function doPost(e) {
  try {
    let payload = {};
    if (e && e.postData && e.postData.contents) {
      payload = JSON.parse(e.postData.contents);
    }
    const serviceUrl = payload.serviceUrl || payload.url;
    if (serviceUrl) {
      const cleanUrl = serviceUrl.trim().replace(/\/+$/, '');
      PropertiesService.getScriptProperties().setProperty(ENV_CLOUD_RUN_URL_KEY, cleanUrl);
      return ContentService.createTextOutput(JSON.stringify({
        status: 'success',
        message: 'Cloud Run URL registered successfully to PropertiesService.',
        url: cleanUrl
      })).setMimeType(ContentService.MimeType.JSON);
    }
    return ContentService.createTextOutput(JSON.stringify({
      status: 'error',
      message: 'No serviceUrl provided in request.'
    })).setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({
      status: 'error',
      message: err.toString()
    })).setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * 백엔드 서버 헬스체크
 */
function checkBackendHealth() {
  const baseUrl = getCloudRunUrl();
  try {
    const res = UrlFetchApp.fetch(baseUrl + '/health', {
      method: 'get',
      muteHttpExceptions: true
    });
    return {
      ok: res.getResponseCode() === 200,
      code: res.getResponseCode(),
      data: JSON.parse(res.getContentText()),
      url: baseUrl
    };
  } catch (e) {
    return {
      ok: false,
      error: e.toString(),
      url: baseUrl
    };
  }
}

/**
 * 미로 생성 API 중계
 */
function generateMazeApi(preset, size) {
  const baseUrl = getCloudRunUrl();
  try {
    const res = UrlFetchApp.fetch(baseUrl + '/api/generate', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        preset: preset || 'recursive_backtracker',
        size: size || 21
      }),
      muteHttpExceptions: true
    });
    return JSON.parse(res.getContentText());
  } catch (e) {
    return {
      status: 'error',
      message: e.toString()
    };
  }
}

/**
 * 미로 BFS vs DFS 탐색 API 중계
 */
function solveMazeApi(maze, start, goal) {
  const baseUrl = getCloudRunUrl();
  try {
    const res = UrlFetchApp.fetch(baseUrl + '/api/solve', {
      method: 'post',
      contentType: 'application/json',
      payload: JSON.stringify({
        maze: maze,
        start: start || { r: 1, c: 1 },
        goal: goal || null
      }),
      muteHttpExceptions: true
    });
    return JSON.parse(res.getContentText());
  } catch (e) {
    return {
      status: 'error',
      message: e.toString()
    };
  }
}
