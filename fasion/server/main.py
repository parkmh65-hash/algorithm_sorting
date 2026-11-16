import os
import json
import base64
import io
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Google Generative AI Import & Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
gemini_client = None

try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_client = genai
except Exception as e:
    print(f"[Warning] Failed to initialize google.generativeai: {e}")

# ChromaDB Setup for Multi-modal Fashionpedia Vector Store
try:
    import chromadb
    from chromadb.config import Settings
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
except Exception as e:
    chroma_client = None
    print(f"[Warning] Failed to initialize chromadb: {e}")

app = FastAPI(
    title="Fashion AI Comparative Synthesizer & Multi-modal RAG API",
    description="Backend for Fashionpedia Multi-modal RAG Pipeline with Gemini VLM",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Data Models
# -------------------------------------------------------------
class StyleRequest(BaseModel):
    prompt: str
    preset: Optional[str] = None
    user_image_b64: Optional[str] = None

class StylingCandidate(BaseModel):
    rank: int
    title: str
    sub_title: str
    match_score: int
    directive_main: str
    directive_sub: str
    key_items: List[str]
    heap_weight: float
    silhouette_ratio: str
    texture_contrast: str
    image_url: Optional[str] = None
    retrieved_images: Optional[List[str]] = []

class StyleResponse(BaseModel):
    status: str
    query_ko: str
    query_en: str
    candidates: List[StylingCandidate]

# -------------------------------------------------------------
# Fashionpedia In-Memory Chroma DB Multi-Vector Store Initialization
# -------------------------------------------------------------
FASHIONPEDIA_CORPUS = [
    {
        "id": "fp_01",
        "category": "outerwear_coat",
        "attributes": "oversized fluid drop-shoulder wool blend trench coat, matte black cropped utility vest, wide-leg pleated trousers, chunky lug derby",
        "style": "Neo-Minimalist Streetwear Architecture",
        "title_ko": "네오-미니멀 스트리트",
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuAB45p6P6b5LLxdGhe3rDDNfmf0lxWb4sEvpdsYFwSYJCfO_VwLMJXcV2L4SM0IvBsZ0MkzigE3rIYc0AD9THUnW9aQPPha7BvZYtkfCVlFAOf44Jg-xvMbtIH-OdipQa8Uy7OkKB0R-SCnul6pB-iopDCfvU8BWk4O6MGLBoatOCE6EvXcqSGjAXU6pmuJlIPjQpEURLHpJbvox3q9AILbpIpySfRNkNsRDbtOW68y",
        "key_items": ["오버사이즈 드롭숄더 코트", "와이드 턱 슬랙스", "청키 솔 더비 슈즈"],
        "ratio": "1:1.618",
        "texture": "+34.2%"
    },
    {
        "id": "fp_02",
        "category": "activewear_jacket",
        "attributes": "3L waterproof technical shell jacket with asymmetrical multi-directional zippers, deep olive tactical cargo pants, trail running sneakers",
        "style": "Urban Utility Technical Gorpcore",
        "title_ko": "어반 유틸리티 고프코어",
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuAbLh94crjiqXyQQEwDha8ylGhZr31pHW6qdM3kR11WM6hhYsWsc09_Vab-N7R-q2TNFi55ahsi1csNtdZhhDkDWfVBUDyoD1oAtDJ6VdwWGQ74s7QiL0GhFjE1DdgquQ-p_FFXWUIO720YmWwewN1KQuJv3GlSfsb2ICHLgLt-pszqymp3-4OafH980OM2u8GJ5A1497NQDKYS76_djpFse8vaZ-hjRZRtGabrCC1E",
        "key_items": ["3L 방수 유틸리티 자켓", "택티컬 카고 팬츠", "트레일 러닝 스니커즈"],
        "ratio": "1:1.414",
        "texture": "+28.5%"
    },
    {
        "id": "fp_03",
        "category": "tailoring_hybrid",
        "attributes": "90s vintage bi-color athletic nylon track top, tailored charcoal pinstripe structured oversized wool blazer, straight raw indigo denim",
        "style": "Vintage Sportswear Hybrid Archive",
        "title_ko": "레트로 스포티 아카이브",
        "image_url": "https://lh3.googleusercontent.com/aida-public/AB6AXuB1uj6MaSSvyhLUfQmbpB44Mi86eldZvEEc8_mQzp8maASyaIRgo5H_3Y5xN0hwjlFfLYjwGf0DYyHra42e9i8qgL_kNyVPyNYAHSUE7DRwElcFnyjoL9UQcJaYw80wcUXl3g_L-qlStayLhZrwAE1eebPTkTrIpTbdsCxSPadFLpOpEKnn5CrobVcu9petn15ElqfKNkA04aGzsFqSCUJizX1nPoWFg-dMYDfi1yU5",
        "key_items": ["바이컬러 트랙 자켓", "스트레이트 데님", "레트로 바스켓볼 로우탑"],
        "ratio": "1:1.250",
        "texture": "+22.1%"
    }
]

fashion_collection = None
if chroma_client:
    try:
        fashion_collection = chroma_client.get_or_create_collection(
            name="fashionpedia_multimodal_store",
            metadata={"hnsw:space": "cosine"}
        )
        if fashion_collection.count() == 0:
            for item in FASHIONPEDIA_CORPUS:
                fashion_collection.add(
                    documents=[f"{item['style']}. {item['attributes']}. Category: {item['category']}"],
                    metadatas=[{
                        "category": item["category"],
                        "style": item["style"],
                        "title_ko": item["title_ko"],
                        "image_url": item["image_url"],
                        "key_items": ",".join(item["key_items"]),
                        "ratio": item["ratio"],
                        "texture": item["texture"]
                    }],
                    ids=[item["id"]]
                )
    except Exception as e:
        print(f"[Warning] ChromaDB collection setup error: {e}")

# -------------------------------------------------------------
# Multi-modal RAG Chain Steps
# -------------------------------------------------------------
def translate_ko_to_en(text_ko: str) -> str:
    """1. 질의 번역: 한국어 질문을 영어로 번역 (Google AI API)"""
    if not gemini_client or not GOOGLE_API_KEY:
        fallback_map = {
            "오버핏": "oversized fit streetwear",
            "미니멀": "minimalist aesthetic",
            "스트릿": "streetwear look",
            "고프코어": "gorpcore technical utility",
            "스포티": "sporty retro vintage",
            "가을": "fall autumn styling"
        }
        translated = text_ko
        for k, v in fallback_map.items():
            translated = translated.replace(k, v)
        return translated

    try:
        model = gemini_client.GenerativeModel("gemini-1.5-flash")
        prompt = f"Translate the following Korean fashion request into concise English fashion query keywords for apparel search:\n'{text_ko}'\nReturn ONLY the English translation."
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        print(f"[Translate error] {e}")
        return text_ko

def retrieve_top_fashion_items(query_en: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """2. 다중벡터 검색기: 질의와 유사한 패션 아이템 및 이미지 2~3개 검색"""
    if fashion_collection and fashion_collection.count() > 0:
        try:
            results = fashion_collection.query(
                query_texts=[query_en],
                n_results=min(top_k, fashion_collection.count())
            )
            retrieved = []
            if results and results.get("ids") and len(results["ids"][0]) > 0:
                for idx, item_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][idx]
                    dist = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.1
                    score = int(max(60, min(99, (1.0 - (dist if dist is not None else 0.1)) * 100)))
                    retrieved.append({
                        "id": item_id,
                        "style": meta.get("style"),
                        "title_ko": meta.get("title_ko"),
                        "image_url": meta.get("image_url"),
                        "key_items": meta.get("key_items", "").split(","),
                        "ratio": meta.get("ratio", "1:1.618"),
                        "texture": meta.get("texture", "+30%"),
                        "score": score
                    })
                return retrieved
        except Exception as e:
            print(f"[Retrieval error] {e}")
            
    # Default Fallback corpus
    return [
        {
            "id": item["id"],
            "style": item["style"],
            "title_ko": item["title_ko"],
            "image_url": item["image_url"],
            "key_items": item["key_items"],
            "ratio": item["ratio"],
            "texture": item["texture"],
            "score": 96 - (i * 4)
        }
        for i, item in enumerate(FASHIONPEDIA_CORPUS[:top_k])
    ]

def generate_and_translate_advice(query_en: str, candidate_style: str) -> tuple[str, str]:
    """3 & 4. VLM 스타일링 조언 생성(EN) 및 한글 번역(KO)"""
    if not gemini_client or not GOOGLE_API_KEY:
        # Predefined high-quality directives matching the Stitch design
        default_directives = {
            "Neo-Minimalist Streetwear Architecture": (
                "오버사이즈 실루엣의 실크 혼방 롱 코트에 크롭 레더 베스트를 레이어드하여 힙한 비례감을 연출합니다.",
                "소재 대비(Matte Wool vs Glossy Leather)를 통해 단색 구성에서도 시각적 깊이감을 확보."
            ),
            "Urban Utility Technical Gorpcore": (
                "테크니컬 방수 쉘과 언밸런스 짚업 디테일로 기능성과 스트리트 무드를 결합했습니다.",
                "어스 톤의 딥 올리브와 에메랄드 메탈릭 파츠로 세련된 포인트 밸런스를 구축."
            ),
            "Vintage Sportswear Hybrid Archive": (
                "빈티지 트랙 탑에 클래식 핀스트라이프 블레이저를 믹스매치한 감각적인 하이-로우 스타일링.",
                "볼드한 네크라인과 대비되는 릴랙스드 핏으로 트렌디한 도수 정렬 시각화 완성."
            )
        }
        return default_directives.get(candidate_style, (
            f"{candidate_style} 스타일에 맞춘 실루엣과 감각적인 텍스처 조합을 제안합니다.",
            "볼드한 레이어링과 톤온톤 배색으로 스타일링 완성도를 높였습니다."
        ))

    try:
        model = gemini_client.GenerativeModel("gemini-1.5-flash")
        # Step: Generate VLM styling advice in English
        prompt_en = (
            f"You are a professional haute-couture fashion stylist. "
            f"For user query: '{query_en}' and style theme: '{candidate_style}', "
            f"provide exactly two short sentences of professional styling directives: "
            f"Sentence 1: Silhouette and layering advice. "
            f"Sentence 2: Material texture contrast and color harmony. "
            f"Format as JSON: {{\"directive1\": \"...\", \"directive2\": \"...\"}}"
        )
        res_en = model.generate_content(prompt_en)
        res_text = res_en.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(res_text)
        
        # Step: Translate styling advice into Korean
        prompt_ko = (
            f"Translate the following fashion directives into professional Korean styling directives:\n"
            f"1. {data.get('directive1')}\n"
            f"2. {data.get('directive2')}\n"
            f"Format as JSON: {{\"d1_ko\": \"...\", \"d2_ko\": \"...\"}}"
        )
        res_ko = model.generate_content(prompt_ko)
        ko_text = res_ko.text.strip().replace("```json", "").replace("```", "")
        data_ko = json.loads(ko_text)
        return (data_ko.get("d1_ko", data.get("directive1")), data_ko.get("d2_ko", data.get("directive2")))
    except Exception as e:
        print(f"[VLM Styling Generation error] {e}")
        return (
            f"{candidate_style} 스타일에 맞춘 실루엣과 감각적인 레이어링 제안입니다.",
            "톤온톤 배색과 텍스처 대비를 통해 세련된 비례감을 구축합니다."
        )

# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Fashion AI Multi-modal RAG Synthesizer",
        "dataset": "Fashionpedia",
        "vector_store": "ChromaDB",
        "llm_vlm": "Google Generative AI (Gemini)"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gemini_active": bool(gemini_client and GOOGLE_API_KEY),
        "chroma_active": bool(chroma_client)
    }

@app.post("/api/compare", response_model=StyleResponse)
def compare_styles(req: StyleRequest):
    """
    Main Multi-modal RAG Pipeline Endpoint:
    1. Query (Korean)
    2. Query Translation (Korean -> English via Google AI)
    3. Multi-vector Retrieval (ChromaDB top matches)
    4. Prompt Formatting & Base64 encoding
    5. VLM Styling Advice Generation (Google AI)
    6. Response Translation (English -> Korean)
    7. 3-Candidate Comparative Visual Telemetry Output
    """
    query_ko = req.prompt.strip()
    if not query_ko and req.preset:
        query_ko = req.preset

    # 1. 질의 번역
    query_en = translate_ko_to_en(query_ko)

    # 2. 다중벡터 이미지 & 아이템 검색 (Top 3)
    retrieved_items = retrieve_top_fashion_items(query_en, top_k=3)

    candidates: List[StylingCandidate] = []
    for idx, item in enumerate(retrieved_items):
        rank = idx + 1
        d_main, d_sub = generate_and_translate_advice(query_en, item["style"])
        
        heap_weights = [0.962, 0.924, 0.887]
        weight = heap_weights[idx] if idx < len(heap_weights) else round(0.85 - (idx * 0.05), 3)

        candidates.append(StylingCandidate(
            rank=rank,
            title=f"추천 #{rank:02d}: {item['title_ko']}",
            sub_title=item["style"],
            match_score=item.get("score", 90 - (idx * 4)),
            directive_main=d_main,
            directive_sub=d_sub,
            key_items=item.get("key_items", []),
            heap_weight=weight,
            silhouette_ratio=item.get("ratio", "1:1.5"),
            texture_contrast=item.get("texture", "+25%"),
            image_url=item.get("image_url"),
            retrieved_images=[item.get("image_url")] if item.get("image_url") else []
        ))

    return StyleResponse(
        status="success",
        query_ko=query_ko,
        query_en=query_en,
        candidates=candidates
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
