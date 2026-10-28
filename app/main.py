import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from qdrant_client import QdrantClient

from app.core.config import settings
from app.services.embedding.fastembed_embedder import FastEmbedEmbedder
from app.services.storage.qdrant_store import QdrantVectorStore
from app.services.retrieval.qdrant_document_retriever import QdrantDocumentRetriever
from app.services.generation.dummy_llm_synthesizer import DummyLLMSynthesizer
from app.services.verification.cross_encoder import CrossEncoderVerifier
from app.services.metrics.prometheus_collector import PrometheusMetricsCollector
from app.orchestrator.cache_orchestrator import CacheOrchestrator
from app.api.routes import router

# -------------------------------------------------------------
# Composition Root (Dependency Injection & SOLID Wiring)
# -------------------------------------------------------------

def build_orchestrator() -> tuple[CacheOrchestrator, list]:
    print("[VeriCache] Initializing SOLID services with REAL document retrieval ($0 cost)...")

    # 1. Local Dense Embedder (ONNX FastEmbed BAAI/bge-small-en-v1.5 on CPU)
    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model_name,
        cache_dir=settings.cache_storage_dir,
        dimension=settings.vector_dimension
    )

    # 2. Shared Qdrant Embedded Engine (stores both cache index and real documents)
    os.makedirs(settings.qdrant_storage_dir, exist_ok=True)
    try:
        qdrant_client = QdrantClient(path=settings.qdrant_storage_dir)
    except Exception as e:
        print(f"[VeriCache] Persistent Qdrant store busy ({e}), utilizing in-memory engine")
        qdrant_client = QdrantClient(":memory:")

    # 3. Vector Cache Store (stores previously served user queries and responses)
    store = QdrantVectorStore(
        client=qdrant_client,
        dimension=settings.vector_dimension
    )

    # 4. REAL Document Knowledge Base Retriever ($0 external cost)
    retriever = QdrantDocumentRetriever(
        client=qdrant_client,
        embedder=embedder,
        dimension=settings.vector_dimension
    )

    # 5. Populate Knowledge Base with REAL Enterprise Policy Document Chunks
    real_policy_documents = [
        {
            "chunk_id": "doc_orders_v1#chunk_02",
            "document_title": "Ecommerce Order Management & Cancellation Policy",
            "content": "To cancel an ecommerce order, navigate to Your Orders within 1 hour of placement and click 'Cancel Items'. After 1 hour, the order moves to dispatch and must instead be returned after delivery."
        },
        {
            "chunk_id": "doc_subscriptions_v1#chunk_01",
            "document_title": "SaaS & VIP Membership Subscription Cancellation Policy",
            "content": "To cancel your recurring membership or subscription: Go to Account Settings > Subscriptions > Select Active Plan > Click 'Cancel Subscription'. Benefits remain active until the end of the billing period; subscriptions cannot be canceled via the Order History page."
        },
        {
            "chunk_id": "doc_returns_v1#chunk_05",
            "document_title": "Footwear & Apparel Return Guidelines",
            "content": "Shoes and footwear may be returned within 45 calendar days of receipt if unworn, with original tags attached and in the original shoebox. Items showing street wear are ineligible for refund."
        },
        {
            "chunk_id": "doc_electronics_v1#chunk_08",
            "document_title": "Consumer Electronics & Laptop Return Policy",
            "content": "Consumer electronics, laptops, and computer hardware must be returned within 14 calendar days of delivery with all serial numbers intact, original packaging present, and personal accounts wiped."
        },
        {
            "chunk_id": "doc_underwriting_v1#chunk_11",
            "document_title": "Personal Loan Underwriting Approval Criteria",
            "content": "Personal loan applications are approved if: FICO credit score meets the minimum tier (720+), debt-to-income (DTI) is below 35%, and applicant has 2+ years of verified steady employment with no recent delinquencies."
        },
        {
            "chunk_id": "doc_underwriting_v1#chunk_12",
            "document_title": "Personal Loan Underwriting Adverse Action & Decline Criteria",
            "content": "Personal loan applications are declined if: Debt-to-income (DTI) exceeds 43%, credit score is below 620, or recent delinquent marks or bankruptcies appear on credit bureau reports."
        },
        {
            "chunk_id": "doc_plans_v1#chunk_01",
            "document_title": "Enterprise Plan & Custom Contract SLA Policy",
            "content": "To upgrade to the Enterprise plan: Contact enterprise-sales@company.com or visit the billing settings to schedule a demo. Enterprise plans include private VPC deployment, 99.99% uptime SLA, and custom volume pricing."
        }
    ]
    retriever.seed_documents(real_policy_documents)
    print(f"[VeriCache] Successfully indexed {len(real_policy_documents)} REAL enterprise policy chunks in Qdrant.")

    # 6. Synthesizer (Dummy/Mock LLM to save paid API fees while grounding on real chunks)
    synthesizer = DummyLLMSynthesizer()

    # 7. Local Cross-Encoder Verifier (ONNX ms-marco-MiniLM-L-6-v2)
    verifier = CrossEncoderVerifier(
        model_name=settings.verifier_model_name,
        cache_dir=settings.cache_storage_dir,
        logit_threshold=settings.verifier_logit_threshold
    )

    # 8. Prometheus Telemetry Collector
    metrics = PrometheusMetricsCollector()

    # 9. Pipeline Flow Orchestrator (DIP)
    orchestrator = CacheOrchestrator(
        embedder=embedder,
        store=store,
        retriever=retriever,
        synthesizer=synthesizer,
        verifier=verifier,
        metrics=metrics,
        naive_cosine_threshold=settings.naive_cosine_threshold
    )

    # 10. Baseline Cached Queries (Seed cache with 4 entries)
    initial_seeds = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "query": "How do I cancel my order?",
            "response": "To cancel an order, navigate to Your Orders within 1 hour of placement and click 'Cancel Items'.",
            "chunk_ids": ["doc_orders_v1#chunk_02"],
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "query": "What is the return window for shoes and footwear?",
            "response": "Shoes can be returned within 45 days of receipt if unworn and in original packaging.",
            "chunk_ids": ["doc_returns_v1#chunk_05"],
        },
        {
            "id": "33333333-3333-3333-3333-333333333333",
            "query": "Why was my personal loan application approved?",
            "response": "Your loan was approved because your credit score met the minimum tier (720+) and debt-to-income was below 35%.",
            "chunk_ids": ["doc_underwriting_v1#chunk_11"],
        },
        {
            "id": "44444444-4444-4444-4444-444444444444",
            "query": "How do I upgrade to the enterprise plan?",
            "response": "To upgrade to Enterprise, contact sales@example.com or visit the billing settings to schedule a demo.",
            "chunk_ids": ["doc_plans_v1#chunk_01"],
        }
    ]

    orchestrator.reset_seeds(initial_seeds)

    # Run realistic benchmark queries so Prometheus and Grafana have initial live data
    demo_queries = [
        "How do I cancel my order?",
        "How do I cancel my subscription?",
        "What is the return window for a laptop?",
        "What is the return window for shoes and footwear?",
        "Why was my personal loan rejected?",
        "Why was my personal loan application approved?",
        "Can I return shoes after 30 days?",
        "Why was my mortgage denied?"
    ]
    for q in demo_queries:
        orchestrator.evaluate_query(q, save_miss_to_cache=False)

    return orchestrator, initial_seeds

# -------------------------------------------------------------
# FastAPI Application Assembly
# -------------------------------------------------------------

app = FastAPI(
    title="VeriCache Control Plane",
    description="Deterministic Semantic Cache with Cross-Encoder Correctness Guarantees, Real Document Retrieval & Prometheus Metrics",
    version="2.0.0"
)

orchestrator, initial_seeds = build_orchestrator()
app.state.orchestrator = orchestrator
app.state.initial_seeds = initial_seeds

# Mount clean API routes
app.include_router(router)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/grafana", response_class=HTMLResponse)
def grafana_page():
    grafana_path = os.path.join(TEMPLATES_DIR, "grafana.html")
    with open(grafana_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
