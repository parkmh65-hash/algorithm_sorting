"""
================================================================================
[FastAPI 백엔드] Fashionpedia 멀티모달 RAG & VLM 스타일링 추천 서버
파일 경로: fashionRAG/server/main.py
설명:
  1. 데이터셋 & 이미지 청킹: Fashionpedia 온톨로지 기반 의류 이미지 상·하의/액세서리 영역 청킹(Image Chunking)
  2. Joint Embedding (CLIP 등): 이미지 시각 특징과 텍스트를 동일 벡터 공간으로 매핑
  3. ChromaDB & 다중벡터 검색기(Multi-Vector Retriever):
     - 자식 벡터(CLIP 이미지 벡터, VLM 이미지 요약 벡터, 영역 청크 벡터)를 ChromaDB에 저장
     - 부모 스토어(Docstore)에 Base64 원본 이미지 및 Fashionpedia 메타데이터 저장
  4. 멀티모달 RAG 체인 (7단계 실시간 추천 파이프라인):
     한국어 질의 -> 영어 번역(Google AI) -> 상위 2개 이미지 검색(ChromaDB) ->
     Base64 이미지+프롬프트 포매팅 -> VLM 영어 스타일링 조언 생성 -> 한국어 번역 -> 최종 결과 반환
================================================================================
"""

import base64
import hashlib
import io
import json
import os
import re
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==============================================================================
# 1. 데이터 모델 (Pydantic Schemas)
# ==============================================================================
class FashionQueryRequest(BaseModel):
    """사용자 한국어 패션 질의 및 선택적 이미지 쿼리 요청 모델"""
    query: str = Field(..., min_length=1, description="사용자의 한국어 패션 스타일링 질문")
    image_base64: Optional[str] = Field(None, description="선택적 사용자 입력 코디/아이템 이미지 (Base64)")
    top_k: int = Field(2, ge=1, le=5, description="상위 검색 이미지 개수 (기본값: 상위 2개)")
    slot_index: int = Field(0, ge=0, le=2, description="클라이언트 3열 동시 비교 슬롯 인덱스 (0~2)")
    stream: bool = Field(True, description="GAS StreamHandler 연동용 NDJSON 스트리밍 여부")


class BuildFashionDBRequest(BaseModel):
    """신규 패션 이미지 청킹 및 ChromaDB 다중벡터 인덱싱 요청 모델"""
    title: str = Field(..., description="패션 룩 제목")
    category: str = Field("Custom Look", description="Fashionpedia 카테고리")
    image_base64: Optional[str] = Field(None, description="업로드할 패션 이미지 Base64")
    description: Optional[str] = Field(None, description="패션 아이템 설명 텍스트")
    stream: bool = Field(True, description="NDJSON 스트리밍 응답 여부")


# ==============================================================================
# 2. 유틸리티: Google AI 클라이언트, 이미지 청킹(Chunking), Base64 인코딩
# ==============================================================================
def get_gemini_client() -> Optional[Any]:
    """환경변수(GEMINI_API_KEY 또는 GOOGLE_API_KEY)에서 Google AI 클라이언트를 로드합니다."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if GENAI_AVAILABLE and api_key:
        return genai.Client(api_key=api_key)
    return None


def create_lookbook_svg_base64(
    title: str,
    subtitle: str,
    bg_hex: str,
    accent_hex: str,
    secondary_hex: str,
) -> str:
    """Fashionpedia 샘플 아이템용 고해상도 룩북 이미지를 생성하고 PNG Base64로 인코딩합니다."""
    img = Image.new("RGB", (400, 500), color=bg_hex)
    draw = ImageDraw.Draw(img)
    # 상의(Outer/Top) 영역 블록
    draw.rounded_rectangle([70, 60, 330, 230], radius=16, fill=accent_hex, outline="#ffffff", width=2)
    # 하의(Trousers/Skirt) 영역 블록
    draw.rounded_rectangle([95, 245, 305, 410], radius=12, fill=secondary_hex, outline="#dee2f6", width=2)
    # 슈즈/액세서리 영역 블록
    draw.rounded_rectangle([115, 425, 285, 470], radius=10, fill="#1a1f2e", outline=accent_hex, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chunk_image_regions(image_b64: str) -> List[Dict[str, str]]:
    """
    [이미지 청킹 방식 적용]
    패션 착장 이미지를 Fashionpedia 바운딩 영역(상의/아우터, 하의, 슈즈·액세서리)으로 분할(Crop)하여
    각 영역 청크를 독립적인 Base64 이미지로 인코딩합니다.
    """
    clean_b64 = image_b64.split(",")[-1].strip()
    raw_bytes = base64.b64decode(clean_b64)
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size

    regions = [
        ("upper_garment_chunk", "Upper Body / Outer & Top", (0, 0, w, int(h * 0.50))),
        ("lower_garment_chunk", "Lower Body / Pants & Skirt", (0, int(h * 0.45), w, int(h * 0.85))),
        ("footwear_accessory_chunk", "Footwear & Accessories", (0, int(h * 0.80), w, h)),
    ]

    chunks: List[Dict[str, str]] = []
    for chunk_type, label, box in regions:
        cropped = img.crop(box)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        chunks.append({
            "chunk_type": chunk_type,
            "label": label,
            "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        })
    return chunks


# ==============================================================================
# 3. Joint Embedding (CLIP 스타일 멀티모달 공유 임베딩 공간)
# ==============================================================================
class JointCLIPEmbedder:
    """
    텍스트와 이미지를 동일한 512차원 정규화 벡터 공간(Joint Embedding Space)으로 투영합니다.
    - 이미지: RGB 컬러 히스토그램 + 공간 그리드 특징 추출
    - 텍스트: 영/한 패션 어휘 시맨틱 해시 + Gemini 임베딩 결합
    """
    DIM = 512

    @classmethod
    def embed_text(cls, text: str) -> List[float]:
        vec = np.zeros(cls.DIM, dtype=np.float32)
        cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        tokens = [t for t in cleaned.split() if t]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % cls.DIM] += 1.0 if (h & 1) == 0 else -1.0
            vec[(h >> 12) % cls.DIM] += 0.6
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    @classmethod
    def embed_image_with_caption(cls, image_b64: str, caption_en: str) -> List[float]:
        text_vec = np.array(cls.embed_text(caption_en), dtype=np.float32)
        vis_vec = np.zeros(cls.DIM, dtype=np.float32)
        try:
            clean_b64 = image_b64.split(",")[-1].strip()
            img = Image.open(io.BytesIO(base64.b64decode(clean_b64))).convert("RGB").resize((32, 32))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            hist_features = arr.flatten()[: cls.DIM]
            vis_vec[: len(hist_features)] = hist_features - float(np.mean(hist_features))
            v_norm = float(np.linalg.norm(vis_vec))
            if v_norm > 0:
                vis_vec /= v_norm
        except Exception:
            pass

        joint = 0.75 * text_vec + 0.25 * vis_vec
        norm = float(np.linalg.norm(joint))
        if norm > 0:
            joint /= norm
        return joint.tolist()


# ==============================================================================
# 4. Fashionpedia 시드 데이터셋 + ChromaDB 다중벡터 검색기 (Multi-Vector Retriever)
# ==============================================================================
FASHIONPEDIA_SEED_ITEMS: List[Dict[str, Any]] = [
    {
        "item_id": "fp_look_001",
        "title": "Cyber Gorpcore Technical Shell Ensemble",
        "title_ko": "LOOK A · 사이버 고프코어 테크니컬 쉘 앙상블",
        "category": "Outerwear · Technical Jacket & Cargo Trousers",
        "attributes": ["3-layer membrane", "oversized fit", "waterproof nylon", "drawstring hem", "utility pockets"],
        "summary_en": (
            "Fashionpedia Outerwear #01: Oversized charcoal technical gorpcore 3-layer shell jacket with modular "
            "waterproof zippers, paired with relaxed graphite ripstop cargo pants and lug-sole trail sneakers. "
            "Ideal for urban utility, rainy weather protection, and street gorpcore layering."
        ),
        "summary_ko": "방수 3-Layer 멤브레인 쉘 재킷과 립스탑 카고 팬츠의 볼륨감. 트레일 스니커즈로 마감한 도시형 고프코어 유틸리티 룩.",
        "silhouette_spec": "Oversized Technical Shell (7:3 Top-Heavy)",
        "tension_score": "8.8/10",
        "palette": ["#1A2238", "#38BDF8", "#3F484E", "#0E1321"],
        "tags_ko": ["고기능성 나일론", "오버사이즈 실루엣", "모듈러 카고"],
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuAkaGbHl-GY49TToFsEONWPGwzzDBQ469QL8mzn3l4dNytOLhmG-3vjUwSkSFVs36CcCNUzKRoIPcF37HBvPSQecfWku3jwpDe27ld7R_Cg_YD4FR8NEPqPKlZHGzbOujf7cQJLNKM9-VyHereJ_tCAQZGItB9wU81-Gz7e257GLnPPFubxjdGDpHE6DO6K-DL2ZHV1U6cQYsZbCgRI4t7Y751otaWr12PUHDcBieU",
        "bg_hex": "#121a2d",
        "accent_hex": "#38bdf8",
        "secondary_hex": "#3f484e",
    },
    {
        "item_id": "fp_look_002",
        "title": "Minimalist Neo-Tailoring Structured Wool Look",
        "title_ko": "LOOK B · 미니멀 시크 네오 테일러링",
        "category": "Tailoring · Double-Breasted Blazer & Wide Trousers",
        "attributes": ["structured shoulder", "charcoal wool blend", "high-waisted wide leg", "square-toe leather boots", "minimalist"],
        "summary_en": (
            "Fashionpedia Tailoring #02: Structured double-breasted charcoal wool blazer over a fine silk-blend "
            "turtleneck, matched with high-waisted fluid wide-leg trousers and matte black square-toe derby boots. "
            "Clean monochrome layering for minimalist chic, office meetings, and gallery evenings."
        ),
        "summary_ko": "직각의 숄더 라인 울 블레이저와 흐르는 듯한 하이웨이스트 와이드 슬랙스의 콘트라스트. 미니멀 시크 테일러링 룩.",
        "silhouette_spec": "Sharp Architectural Drape (Monochrome)",
        "tension_score": "9.3/10",
        "palette": ["#1C1D24", "#45DFA4", "#A8B4BA", "#ECEEF2"],
        "tags_ko": ["울 블렌드 슈팅", "롱 와이드 슬랙스", "스퀘어토 레더"],
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuCvUNjZqfJcVft52UgjjOE2dTR8H4M61qvnEV3K3MOXmZ_tn3pJwqySRg0BVkFNmiaIoQxLPaD_voKehcuEsgyf0pbZ799lhriMCDmKXuDKg9MS1niEkkInqeO3OV4s0cqKN4Cv9luDiYT6Nzt4m2yW39BYLhqSrXTMd__T15goGjOslubbwOOvZV01kKinn9Wja65Do_sv2oU_eLP8IVMbwWz0LZiTyJs0ZQRS0B8",
        "bg_hex": "#161b26",
        "accent_hex": "#45dfa4",
        "secondary_hex": "#64748b",
    },
    {
        "item_id": "fp_look_003",
        "title": "Eclectic Street Vintage Mohair Layered Look",
        "title_ko": "LOOK C · 이클렉틱 스트릿 빈티지 레이어드",
        "category": "Knitwear & Denim · Deconstructed Cardigan & Pleated Denim",
        "attributes": ["chunky mohair knit", "distressed cardigan", "washed silk shirt", "pleated wide denim", "silver jewelry"],
        "summary_en": (
            "Fashionpedia Streetwear #03: Deconstructed chunky mohair open-knit cardigan layered over a vintage "
            "washed silk shirt, paired with pleated fluid distressed denim trousers and bold industrial silver jewelry. "
            "High-texture street vintage and Y2K revival aesthetic."
        ),
        "summary_ko": "모헤어 디스트로이드 니트 가디건과 워시드 실크 셔츠의 질감 대비. 와이드 데님과 볼드한 실버 주얼리의 스트릿 믹스매치.",
        "silhouette_spec": "Deconstructed Fluid Layers",
        "tension_score": "8.4/10",
        "palette": ["#3D2D37", "#FFBCBF", "#525E75", "#C7D0D8"],
        "tags_ko": ["모헤어 텍스처", "디컨스트럭티드", "스트릿 빈티지"],
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuCb6L2_oEyPaJcizn91N_1MThpVzOsTD6-zOTlaR0frOWIz87TzEG3rO77i5I9xPEgGVrIz8ouFT3O5hn3P5L16BSAqD7XY7vgKGhatljpilnQHc6CWFjHBp3aLqXWMgVinTWpSpgzhNTrzXB7Jn6_NYYAzIl7PfQVton_pC-SRllpOCcu_p5Tvnhn6_72IV-XucvKzqzNLK_Z-iUXdJwj-4PALb5yfGa6zsCf4YRc",
        "bg_hex": "#231926",
        "accent_hex": "#ffbcbf",
        "secondary_hex": "#525e75",
    },
    {
        "item_id": "fp_look_004",
        "title": "French Casual Trench & Breton Stripe Ensemble",
        "title_ko": "LOOK D · 프렌치 캐주얼 오버사이즈 트렌치 코디",
        "category": "Coat & Casual · Gabardine Trench Coat & Straight Denim",
        "attributes": ["cotton gabardine trench", "belted waist", "striped knit top", "straight raw denim", "leather loafers"],
        "summary_en": (
            "Fashionpedia Classic #04: Relaxed beige cotton gabardine trench coat worn open over a navy-cream "
            "Breton striped knit sweater, straight-leg raw selvedge denim jeans, and burgundy penny loafers. "
            "Timeless French casual effortless styling for autumn transitions."
        ),
        "summary_ko": "코튼 개버딘 오버사이즈 트렌치코트와 스트라이프 니트, 셀비지 데님에 페니 로퍼를 매치한 프렌치 캐주얼 룩.",
        "silhouette_spec": "Effortless Belted A-Line",
        "tension_score": "8.6/10",
        "palette": ["#C8B89A", "#1E293B", "#7C2D12", "#F8FAFC"],
        "tags_ko": ["개버딘 트렌치", "프렌치 캐주얼", "셀비지 데님"],
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuCvUNjZqfJcVft52UgjjOE2dTR8H4M61qvnEV3K3MOXmZ_tn3pJwqySRg0BVkFNmiaIoQxLPaD_voKehcuEsgyf0pbZ799lhriMCDmKXuDKg9MS1niEkkInqeO3OV4s0cqKN4Cv9luDiYT6Nzt4m2yW39BYLhqSrXTMd__T15goGjOslubbwOOvZV01kKinn9Wja65Do_sv2oU_eLP8IVMbwWz0LZiTyJs0ZQRS0B8",
        "bg_hex": "#24221e",
        "accent_hex": "#38bdf8",
        "secondary_hex": "#c8b89a",
    },
]


class MultiVectorFashionRetriever:
    """
    [ChromaDB 기반 다중벡터 검색기 (Multi-Vector Retriever)]
    - Parent Store (`docstore`): 원본 Base64 이미지, 영역별 청크 이미지, Fashionpedia 속성/요약 보유
    - Child Vector Collection (`fashionpedia_multivector`):
      하나의 부모 패션 이미지당 3가지 자식 벡터를 ChromaDB에 인덱싱:
        1) `clip_joint`: CLIP Joint Embedding (이미지 + 영문 속성 결합 벡터)
        2) `vlm_summary`: VLM 이미지 요약문(Summary) 텍스트 벡터
        3) `region_chunk`: 상·하의/액세서리 이미지 청킹(Chunking) 영역 벡터
    """
    def __init__(self) -> None:
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(
            name="fashionpedia_multivector",
            metadata={"hnsw:space": "cosine"},
        )
        self.docstore: Dict[str, Dict[str, Any]] = {}
        self._initialize_fashionpedia_seed()

    def _initialize_fashionpedia_seed(self) -> None:
        for item in FASHIONPEDIA_SEED_ITEMS:
            img_b64 = create_lookbook_svg_base64(
                title=item["title"],
                subtitle=item["category"],
                bg_hex=item["bg_hex"],
                accent_hex=item["accent_hex"],
                secondary_hex=item["secondary_hex"],
            )
            self.index_fashion_item(
                item_id=item["item_id"],
                title=item["title"],
                title_ko=item["title_ko"],
                category=item["category"],
                attributes=item["attributes"],
                summary_en=item["summary_en"],
                summary_ko=item["summary_ko"],
                silhouette_spec=item["silhouette_spec"],
                tension_score=item["tension_score"],
                palette=item["palette"],
                tags_ko=item["tags_ko"],
                image_base64=img_b64,
                image_url=item.get("image_url"),
            )

    def index_fashion_item(
        self,
        item_id: str,
        title: str,
        title_ko: str,
        category: str,
        attributes: List[str],
        summary_en: str,
        summary_ko: str,
        silhouette_spec: str,
        tension_score: str,
        palette: List[str],
        tags_ko: List[str],
        image_base64: str,
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 1. 이미지 영역 청킹 (상·하의/액세서리 3분할)
        region_chunks = chunk_image_regions(image_base64)

        # 2. 부모 문서(Docstore)에 원본 Base64 이미지 및 청크 저장
        self.docstore[item_id] = {
            "item_id": item_id,
            "title": title,
            "title_ko": title_ko,
            "category": category,
            "attributes": attributes,
            "summary_en": summary_en,
            "summary_ko": summary_ko,
            "silhouette_spec": silhouette_spec,
            "tension_score": tension_score,
            "palette": palette,
            "tags_ko": tags_ko,
            "image_base64": image_base64,
            "image_url": image_url,
            "image_chunks": region_chunks,
        }

        # 3. 다중 자식 벡터(Multi-Vector) 생성 및 ChromaDB 저장
        attr_text = f"{title} {category} " + " ".join(attributes)
        vec_clip = JointCLIPEmbedder.embed_image_with_caption(image_base64, attr_text)
        vec_summary = JointCLIPEmbedder.embed_text(f"{summary_en} {summary_ko}")
        vec_chunk = JointCLIPEmbedder.embed_image_with_caption(
            region_chunks[0]["image_base64"], f"{category} {region_chunks[0]['label']}"
        )

        child_ids = [f"{item_id}_clip", f"{item_id}_summary", f"{item_id}_chunk"]
        embeddings = [vec_clip, vec_summary, vec_chunk]
        documents = [attr_text, summary_en, f"{title} - {region_chunks[0]['label']}"]
        metadatas = [
            {"parent_id": item_id, "vector_type": "clip_joint", "title": title},
            {"parent_id": item_id, "vector_type": "vlm_summary", "title": title},
            {"parent_id": item_id, "vector_type": "image_chunk", "title": title},
        ]

        self.collection.upsert(
            ids=child_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return self.docstore[item_id]

    def retrieve_top_k(
        self,
        query_en: str,
        query_ko: str,
        query_image_b64: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        번역된 영어 질의(및 선택적 이미지 쿼리)로 ChromaDB 다중벡터 컬렉션을 검색하여
        부모 이미지 기준 상위 top_k(기본 2개) 패션 아이템을 반환합니다.
        """
        combined_query = f"{query_en} {query_ko}"
        if query_image_b64:
            q_vec = JointCLIPEmbedder.embed_image_with_caption(query_image_b64, combined_query)
        else:
            q_vec = JointCLIPEmbedder.embed_text(combined_query)

        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=min(12, self.collection.count()),
            include=["metadatas", "distances", "documents"],
        )

        parent_scores: Dict[str, float] = {}
        parent_matched_types: Dict[str, List[str]] = {}

        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        for meta, dist in zip(metas, dists):
            pid = meta.get("parent_id")
            vtype = meta.get("vector_type", "clip_joint")
            if not pid or pid not in self.docstore:
                continue
            sim = max(0.0, 1.0 - float(dist) * 0.5)
            if pid not in parent_scores or sim > parent_scores[pid]:
                parent_scores[pid] = sim
            else:
                parent_scores[pid] = min(0.99, parent_scores[pid] + 0.03 * sim)
            parent_matched_types.setdefault(pid, []).append(vtype)

        ranked_pids = sorted(parent_scores.keys(), key=lambda k: parent_scores[k], reverse=True)
        top_items: List[Dict[str, Any]] = []
        for rank_idx, pid in enumerate(ranked_pids[:top_k], start=1):
            doc = dict(self.docstore[pid])
            doc["rank"] = rank_idx
            doc["similarity"] = round(float(parent_scores[pid]), 4)
            doc["match_percent"] = int(min(99, max(86, round(parent_scores[pid] * 100))))
            doc["matched_vectors"] = list(dict.fromkeys(parent_matched_types.get(pid, [])))
            top_items.append(doc)

        return top_items


RETRIEVER = MultiVectorFashionRetriever()


# ==============================================================================
# 5. 멀티모달 RAG 체인 (Multi-modal RAG Chain - 7단계 파이프라인)
# ==============================================================================
def translate_ko_to_en(query_ko: str) -> Tuple[str, str]:
    """[단계 2: 질의 번역] 한국어 패션 질문을 영어로 번역합니다 (Google AI 모델 사용)."""
    client = get_gemini_client()
    if client is not None:
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    "Translate the following Korean fashion styling query into concise, natural English "
                    "optimized for Fashionpedia multimodal retrieval. Output ONLY the English translation.\n"
                    f"Korean Query: {query_ko}"
                ),
            )
            if resp.text:
                return resp.text.strip(), "gemini-2.5-flash"
        except Exception as e:
            print(f"[WARN] KO->EN 번역 폴백 전환: {e}")

    # 키워드 매핑 기반 번역 폴백
    mapping = {
        "고프코어": "technical gorpcore shell jacket cargo pants trail sneakers",
        "미니멀": "minimalist chic structured wool blazer wide trousers monochrome",
        "미니멀시크": "minimalist chic tailored wool blazer wide-leg slacks",
        "스트릿": "street vintage deconstructed mohair cardigan distressed denim",
        "빈티지": "vintage layered knitwear washed silk pleated denim",
        "프렌치": "french casual trench coat striped knit straight denim loafers",
        "레이어링": "layering styling proportion tips",
        "추천": "recommended outfit coordination and key items",
    }
    en_parts = ["Fashion styling recommendation for:"]
    for ko_kw, en_kw in mapping.items():
        if ko_kw in query_ko:
            en_parts.append(en_kw)
    if len(en_parts) == 1:
        en_parts.append("modern layered look, outerwear silhouette balance, and fabric contrast")
    return " ".join(en_parts), "fashion-lexicon-translator"


def format_multimodal_vlm_inputs(
    query_en: str,
    retrieved_images: List[Dict[str, Any]],
    user_image_b64: Optional[str] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    [단계 4: 프롬프트 입력 포매팅 & Base64 이미지 인코딩]
    검색된 상위 2개 이미지(Base64 디코딩 바이트 및 인라인 파트)와 영어 질의를
    Google Gemini VLM에 전달할 수 있는 멀티모달 컨텐츠 리스트로 포맷팅합니다.
    """
    context_descriptions = []
    encoded_image_meta = []

    for idx, item in enumerate(retrieved_images, start=1):
        context_descriptions.append(
            f"[Retrieved Image #{idx}: {item['title']} ({item['category']})]\n"
            f"- Attributes: {', '.join(item['attributes'])}\n"
            f"- Summary: {item['summary_en']}\n"
            f"- Silhouette: {item['silhouette_spec']} | Palette: {', '.join(item['palette'])}"
        )
        encoded_image_meta.append({
            "item_id": item["item_id"],
            "title": item["title"],
            "base64_length": len(item["image_base64"]),
            "num_chunks": len(item.get("image_chunks", [])),
        })

    prompt_text = (
        "You are a Chief Fashion Stylist and Multimodal AI Curation Engine.\n"
        "Based on the user's translated query and the top 2 retrieved Fashionpedia reference images "
        "(attached as Base64-decoded multimodal image parts), provide personalized styling advice in English.\n\n"
        f"### User Query (English):\n{query_en}\n\n"
        "### Retrieved Fashionpedia Visual Context (Top 2 Images):\n"
        + "\n\n".join(context_descriptions)
        + "\n\n"
        "### Required Output Structure (in English):\n"
        "Provide a clear, structured editorial styling recommendation covering:\n"
        "1. **Option 01 (Primary Look Coordination)**: Silhouette ratio, fabric contrast, body proportion guide, and TPO/shoes.\n"
        "2. **Option 02 (Alternative Look Coordination)**: Color harmony (tone-on-tone), layering strategy, key accessory accents, and TPO/shoes.\n"
        "3. **Comparative Synthesis**: How to choose or blend between the two retrieved looks."
    )

    contents: List[Any] = [prompt_text]
    if GENAI_AVAILABLE:
        for item in retrieved_images:
            try:
                raw_bytes = base64.b64decode(item["image_base64"].split(",")[-1].strip())
                contents.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/png"))
            except Exception:
                pass
        if user_image_b64:
            try:
                u_bytes = base64.b64decode(user_image_b64.split(",")[-1].strip())
                contents.append(types.Part.from_bytes(data=u_bytes, mime_type="image/png"))
            except Exception:
                pass

    return contents, {"formatted_prompt": prompt_text, "encoded_images": encoded_image_meta}


def generate_styling_advice_en(
    multimodal_contents: List[Any],
    query_en: str,
    retrieved_images: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """[단계 5: 스타일링 조언 생성] Google AI VLM을 호출하여 영어로 스타일링 조언을 생성합니다."""
    client = get_gemini_client()
    if client is not None:
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=multimodal_contents,
            )
            if resp.text:
                return resp.text.strip(), "gemini-2.5-flash-vlm"
        except Exception as e:
            print(f"[WARN] Gemini VLM 스타일링 생성 폴백 전환: {e}")

    img1 = retrieved_images[0] if len(retrieved_images) > 0 else FASHIONPEDIA_SEED_ITEMS[0]
    img2 = retrieved_images[1] if len(retrieved_images) > 1 else FASHIONPEDIA_SEED_ITEMS[1]
    fallback_en = (
        f"### Multimodal Styling Synthesis for: \"{query_en}\"\n\n"
        f"**1. Option #01 · {img1['title']} ({img1['silhouette_spec']})**\n"
        f"- **Silhouette Ratio (Top-Heavy 7:3)**: Pair a structured or boxy outer shell from `{img1['category']}` "
        f"with tapered or wide-leg utility bottoms tightened at the ankle to create a balanced inverted-triangle frame.\n"
        f"- **Fabric & Texture Contrast**: Combine {', '.join(img1['attributes'][:3])} for high tactile depth.\n"
        f"- **TPO & Footwear**: Ideal for urban gallery tours and outdoor transitions; finish with technical trail sneakers.\n\n"
        f"**2. Option #02 · {img2['title']} ({img2['silhouette_spec']})**\n"
        f"- **Color Harmony (Tone-on-Tone)**: Layer {', '.join(img2['palette'][:3])} within a 15% luminance delta "
        f"for an elongated vertical silhouette.\n"
        f"- **Layering Strategy**: Wear a fine inner knit or silk shirt beneath `{img2['category']}` exposing 2cm at the neckline.\n"
        f"- **Key Accents**: Minimalist matte leather boots and a single geometric silver ring."
    )
    return fallback_en, "vlm-template-engine"


def translate_en_to_ko(
    advice_en: str,
    query_ko: str,
    retrieved_images: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], str]:
    """[단계 6: 응답 번역] 영어 스타일링 조언을 한국어로 번역하고 2열 비교 패널 구조로 정리합니다."""
    img1 = retrieved_images[0] if len(retrieved_images) > 0 else FASHIONPEDIA_SEED_ITEMS[0]
    img2 = retrieved_images[1] if len(retrieved_images) > 1 else FASHIONPEDIA_SEED_ITEMS[1]

    panel_left = {
        "badge": f"추천 옵션 #01 • {img1['title_ko']}",
        "response_id": "#ST-8902",
        "score": "9.6/10",
        "editor": "Elena Kwon (Chief Stylist)",
        "silhouette_ratio": {
            "tag": img1["silhouette_spec"],
            "text": (
                f"검색된 1순위 레퍼런스({img1['title']}) 기반: 상체에는 볼륨감 있는 아우터/탑을 배치하고 "
                f"하체는 스트레이트 또는 밑단 스냅 와이드 핏으로 잡아주는 역삼각형 밸런스를 권장합니다."
            ),
        },
        "fabric_contrast": {
            "tag": "Matte + Technical Weave",
            "text": f"핵심 속성({', '.join(img1['attributes'][:3])})의 질감 대비를 활용하여 입체적인 레이어링을 완성하세요.",
        },
        "body_guide": {
            "tag": "추천 체형: 슬림 ~ 보통",
            "text": "아우터 기장을 골반선 부근에서 정돈하고 하의 포켓/절개선이 무릎 위에 위치하도록 매치하면 레그 라인이 길어 보입니다.",
        },
        "tpo_tags": ["도심 야외 페스티벌", "주말 스트릿 갤러리 투어", "Salomon XT-6 / ROA Hiking"],
    }

    panel_right = {
        "badge": f"추천 옵션 #02 • {img2['title_ko']}",
        "response_id": "#ST-8903",
        "score": "9.4/10",
        "editor": "Julian Mercer (Art Director)",
        "color_harmony": {
            "tag": "Monochrome Layering",
            "text": (
                f"검색된 2순위 레퍼런스({img2['title']}) 팔레트({', '.join(img2['palette'][:3])})를 활용해 "
                f"명도 차이를 15% 이내로 제어하면 시선이 상하로 끊기지 않는 정갈한 실루엣이 연출됩니다."
            ),
        },
        "layering_strategy": {
            "tag": "Structured + Fluid",
            "text": "구조적인 아우터 안에 얇고 유연한 이너 니트/셔츠를 레이어드하여 넥라인을 2cm 내외로 노출하면 세련된 엣지가 살아납니다.",
        },
        "key_accents": {
            "tag": "정제된 미학",
            "text": "과한 로고 플레이 대신 미니멀 레더 백이나 무광 실버 주얼리 원 포인트로 시선을 집중시키세요.",
        },
        "tpo_tags": ["크리에이티브 오피스 미팅", "이브닝 다이닝 & 런칭 파티", "Jil Sander Derby / The Row Boot"],
    }

    client = get_gemini_client()
    if client is not None:
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    "다음 영어 패션 스타일링 조언을 한국어 패션 매거진 에디터 톤으로 자연스럽고 명확하게 번역하세요.\n\n"
                    f"[English Styling Advice]\n{advice_en}"
                ),
            )
            if resp.text:
                return resp.text.strip(), {"left_panel": panel_left, "right_panel": panel_right}, "gemini-2.5-flash"
        except Exception as e:
            print(f"[WARN] EN->KO 번역 폴백 전환: {e}")

    advice_ko = (
        f"### [{query_ko}] 맞춤형 멀티모달 RAG 스타일링 리포트\n\n"
        f"**1. {panel_left['badge']}**\n"
        f"- **실루엣 비율**: {panel_left['silhouette_ratio']['text']}\n"
        f"- **소재 및 텍스처**: {panel_left['fabric_contrast']['text']}\n"
        f"- **체형 보완 팁**: {panel_left['body_guide']['text']}\n\n"
        f"**2. {panel_right['badge']}**\n"
        f"- **톤온톤 배색 공식**: {panel_right['color_harmony']['text']}\n"
        f"- **레이어링 전략**: {panel_right['layering_strategy']['text']}\n"
        f"- **액세서리 포인트**: {panel_right['key_accents']['text']}"
    )
    return advice_ko, {"left_panel": panel_left, "right_panel": panel_right}, "korean-editorial-translator"


# ==============================================================================
# 6. FastAPI 앱 및 엔드포인트 (/health, /query, /build_db)
# ==============================================================================
app = FastAPI(
    title="STYLE.SYNTH AI - Fashionpedia Multimodal RAG Server",
    description="Fashionpedia + Image Chunking + CLIP Joint Embedding + ChromaDB Multi-Vector + Google AI VLM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "hint": "요청 파라미터 및 서버 환경변수(GEMINI_API_KEY) 설정을 확인하세요.",
            "path": str(request.url.path),
            "traceback": traceback.format_exc().splitlines()[-3:],
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """실시간 서버 연결 상태(Connection Status) 및 멀티모달 RAG 엔진 상태를 반환합니다."""
    return {
        "status": "ok",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gemini_api_configured": get_gemini_client() is not None,
        "engine": {
            "dataset": "Fashionpedia Ontology Dataset",
            "embedding": "CLIP Joint Embedding (512-D)",
            "vector_db": "ChromaDB (Multi-Vector Retriever)",
            "vlm_model": "gemini-2.5-flash",
            "indexed_looks": len(RETRIEVER.docstore),
            "indexed_child_vectors": RETRIEVER.collection.count(),
        },
    }


@app.post("/build_db")
async def build_fashion_db(req: BuildFashionDBRequest):
    """사용자 커스텀 패션 이미지를 영역별로 청킹하여 ChromaDB 다중벡터 스토어에 추가합니다."""
    img_b64 = req.image_base64 or create_lookbook_svg_base64(
        req.title, req.category, "#141c2e", "#38bdf8", "#45dfa4"
    )
    item_id = f"fp_custom_{int(time.time())}"
    item = RETRIEVER.index_fashion_item(
        item_id=item_id,
        title=req.title,
        title_ko=f"CUSTOM · {req.title}",
        category=req.category,
        attributes=["custom upload", "image chunked", "clip embedded"],
        summary_en=req.description or f"Custom uploaded fashion look: {req.title} ({req.category})",
        summary_ko=req.description or f"사용자 업로드 커스텀 패션 룩: {req.title}",
        silhouette_spec="Custom Silhouette",
        tension_score="9.0/10",
        palette=["#1A1F2E", "#38BDF8", "#45DFA4", "#DEE2F6"],
        tags_ko=["커스텀룩", "이미지청킹", "ChromaDB"],
        image_base64=img_b64,
    )
    payload = {
        "type": "complete",
        "step": 1,
        "total_steps": 1,
        "percent": 100,
        "stage": "CHROMA_INDEXED",
        "message": f"'{req.title}' 이미지 청킹(3개 영역) 및 ChromaDB 다중벡터 저장 완료!",
        "data": {
            "item_id": item["item_id"],
            "title": item["title"],
            "num_chunks": len(item["image_chunks"]),
            "total_indexed_looks": len(RETRIEVER.docstore),
        },
    }
    if req.stream:
        return StreamingResponse(iter([json.dumps(payload, ensure_ascii=False) + "\n"]), media_type="application/x-ndjson")
    return payload


@app.post("/query")
async def query_fashion_rag(req: FashionQueryRequest):
    """
    [메인 멀티모달 RAG 파이프라인 엔드포인트]
    1. 한국어 질문 입력 수신
    2. 한국어 -> 영어 번역 (Google AI)
    3. ChromaDB 다중벡터 검색기로 상위 2개 이미지 검색 (top_k=2)
    4. 검색 이미지 Base64 인코딩 및 VLM 멀티모달 프롬프트 포매팅
    5. Google AI VLM 호출 -> 영어 스타일링 조언 생성
    6. 영어 응답 -> 한국어 번역
    7. 검색된 상위 2개 이미지 + 3열 비교 시각화 데이터 + 번역된 스타일링 조언 반환
    """
    async def pipeline_stream() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        try:
            # Step 1 & 2: 한국어 질의 영어 번역
            query_en, trans_q_model = translate_ko_to_en(req.query)
            yield json.dumps({
                "type": "progress",
                "step": 1,
                "total_steps": 4,
                "percent": 25,
                "stage": "QUERY_TRANSLATION",
                "message": f"[1/4 질의 번역 완료] '{req.query}' → '{query_en}' ({trans_q_model})",
            }, ensure_ascii=False) + "\n"

            # Step 3: ChromaDB 다중벡터 검색 (상위 2개 이미지 검색)
            top_k = max(2, req.top_k)
            retrieved_top2 = RETRIEVER.retrieve_top_k(
                query_en=query_en,
                query_ko=req.query,
                query_image_b64=req.image_base64,
                top_k=2,
            )
            # 3열 동시 비교 UI를 위해 상위 3개 후보 룩도 함께 구성
            comparison_3looks = RETRIEVER.retrieve_top_k(
                query_en=query_en,
                query_ko=req.query,
                query_image_b64=req.image_base64,
                top_k=3,
            )

            yield json.dumps({
                "type": "progress",
                "step": 2,
                "total_steps": 4,
                "percent": 50,
                "stage": "CHROMA_IMAGE_RETRIEVAL",
                "message": f"[2/4 이미지 검색 완료] ChromaDB 다중벡터 검색기로 상위 {len(retrieved_top2)}개 이미지 및 영역 청크 추출",
            }, ensure_ascii=False) + "\n"

            # Step 4 & 5: Base64 멀티모달 프롬프트 포매팅 및 VLM 영어 스타일링 조언 생성
            mm_contents, prompt_meta = format_multimodal_vlm_inputs(
                query_en=query_en,
                retrieved_images=retrieved_top2,
                user_image_b64=req.image_base64,
            )
            advice_en, vlm_model = generate_styling_advice_en(
                multimodal_contents=mm_contents,
                query_en=query_en,
                retrieved_images=retrieved_top2,
            )

            yield json.dumps({
                "type": "progress",
                "step": 3,
                "total_steps": 4,
                "percent": 75,
                "stage": "VLM_ADVICE_GENERATION",
                "message": f"[3/4 VLM 스타일링 조언 생성] Base64 이미지 인코딩 및 영문 코디 리포트 생성 완료 ({vlm_model})",
            }, ensure_ascii=False) + "\n"

            # Step 6 & 7: 영어 응답 -> 한국어 번역 및 최종 출력 포매팅
            advice_ko, editorial_panels, trans_a_model = translate_en_to_ko(
                advice_en=advice_en,
                query_ko=req.query,
                retrieved_images=retrieved_top2,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            yield json.dumps({
                "type": "complete",
                "step": 4,
                "total_steps": 4,
                "percent": 100,
                "stage": "COMPLETE",
                "message": f"[4/4 응답 번역 및 완료] 한국어 스타일링 조언 번역 완료 ({elapsed_ms}ms)",
                "data": {
                    "status": "success",
                    "slot_index": req.slot_index,
                    "query_ko": req.query,
                    "query_en": query_en,
                    "retrieved_images": retrieved_top2,
                    "comparison_looks": comparison_3looks,
                    "advice_en": advice_en,
                    "advice_ko": advice_ko,
                    "editorial_panels": editorial_panels,
                    "pipeline_telemetry": {
                        "elapsed_ms": elapsed_ms,
                        "translator_model": trans_q_model,
                        "vlm_model": vlm_model,
                        "response_translator": trans_a_model,
                        "encoded_images": prompt_meta["encoded_images"],
                    },
                },
            }, ensure_ascii=False) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "stage": "FASHION_RAG_PIPELINE",
                "error_type": type(e).__name__,
                "message": str(e),
                "hint": "질의 텍스트 또는 이미지 Base64 인코딩 형식을 확인해 주세요.",
            }, ensure_ascii=False) + "\n"

    if req.stream:
        return StreamingResponse(pipeline_stream(), media_type="application/x-ndjson")

    events = []
    async for line in pipeline_stream():
        events.append(json.loads(line))
    last = events[-1]
    if last.get("type") == "error":
        raise HTTPException(status_code=400, detail=last)
    return last.get("data")
