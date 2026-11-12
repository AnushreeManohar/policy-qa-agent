import io

import pytest
from conftest import ADVERSARIAL, POLICIES
from pypdf import PdfWriter
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from policyqa.embeddings import HashingEmbedder
from policyqa.ingestion.chunking import chunk_pages, heading_case, is_heading
from policyqa.ingestion.pdf import PdfValidationError, extract_pdf
from policyqa.ingestion.pipeline import infer_policy_type, ingest_bytes, safe_filename
from policyqa.models import PageText
from policyqa.store.memory import MemoryStore


def _pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER, invariant=1)
    for text in pages:
        y = 720
        for line in text.splitlines():
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------- PDF safety
def test_rejects_empty_file():
    with pytest.raises(PdfValidationError, match="empty"):
        extract_pdf(b"", max_pages=10, max_chars_per_page=1000)


def test_rejects_non_pdf_bytes():
    with pytest.raises(PdfValidationError, match="not a PDF"):
        extract_pdf(b"MZ\x90\x00 definitely an executable", max_pages=10, max_chars_per_page=1000)


def test_rejects_truncated_pdf():
    data = POLICIES[0].read_bytes()[:400]
    with pytest.raises(PdfValidationError):
        extract_pdf(data, max_pages=10, max_chars_per_page=1000)


def test_rejects_encrypted_pdf():
    w = PdfWriter()
    w.add_blank_page(612, 792)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(PdfValidationError, match="Encrypted"):
        extract_pdf(buf.getvalue(), max_pages=10, max_chars_per_page=1000)


def test_rejects_pdf_without_text():
    w = PdfWriter()
    w.add_blank_page(612, 792)
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(PdfValidationError, match="no extractable text"):
        extract_pdf(buf.getvalue(), max_pages=10, max_chars_per_page=1000)


def test_rejects_too_many_pages():
    with pytest.raises(PdfValidationError, match="limit"):
        extract_pdf(_pdf(["a"] * 4), max_pages=3, max_chars_per_page=1000)


def test_extracts_pages_and_metadata_title():
    pages, title = extract_pdf(POLICIES[0].read_bytes(), max_pages=50, max_chars_per_page=20_000)
    assert len(pages) >= 3 and [p.page for p in pages] == list(range(1, len(pages) + 1))
    assert title and "Northwind" in title
    assert all(p.text for p in pages)


def test_safe_filename_strips_paths_and_odd_characters():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my policy<script>.pdf") == "my policy_script_.pdf"
    assert safe_filename("") == "document.pdf"


def test_policy_type_inference():
    assert infer_policy_type("x.pdf", "Homeowners Policy (HO-3)") == "homeowners"
    assert infer_policy_type("auto_policy.pdf", "") == "auto"
    assert infer_policy_type("unknown.pdf", "Something") == "general"


# ---------------------------------------------------------------- chunking
@pytest.mark.parametrize("line,expected", [
    ("SECTION II - LIABILITY COVERAGES", True), ("EXCLUSIONS", True), ("4.2 Waiting Periods", True),
    ("We do not insure for loss caused by flood.", False), ("Coverage A - Dwelling limit: $350,000.", False),
    ("Page 3", False),
])
def test_heading_detection(line, expected):
    assert is_heading(line) is expected


def test_heading_case_keeps_roman_numerals():
    assert heading_case("SECTION II - LIABILITY COVERAGES") == "Section II - Liability Coverages"
    assert heading_case("PART D - COVERAGE FOR DAMAGE TO YOUR AUTO") == "Part D - Coverage for Damage to your Auto"


def test_chunks_follow_sections_and_keep_pages():
    pages = [PageText(1, "EXCLUSIONS\nFlood is excluded. Earthquake is excluded.\n"
                            "DEDUCTIBLES\nThe deductible is $500."),
             PageText(2, "Page 2\nCLAIMS\nReport claims within 30 days.")]
    chunks = chunk_pages("d" * 64, pages, size=500, overlap=100)
    assert [c.section for c in chunks] == ["Exclusions", "Deductibles", "Claims"]
    assert [c.page_start for c in chunks] == [1, 1, 2]
    assert all("Page 2" not in c.text for c in chunks)  # running footers are dropped
    assert chunks[0].chunk_id == f"{'d' * 16}:0000"


def test_long_sections_split_on_sentences_with_overlap_and_no_tiny_tail():
    sents = [f"Sentence number {i} explains a separate policy rule in detail." for i in range(40)]
    chunks = chunk_pages("e" * 64, [PageText(1, "CONDITIONS\n" + " ".join(sents))], size=400, overlap=120)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 * 1.5 for c in chunks)
    assert min(len(c.text) for c in chunks) > 400 * 0.3
    for a, b in zip(chunks, chunks[1:]):
        last = a.text.split(". ")[-1].rstrip(".")
        assert last in b.text  # sentence-level overlap carried into the next chunk
    for c in chunks:  # no chunk starts mid-word
        assert c.text[0].isupper()


def test_injection_text_is_flagged_at_ingestion():
    pages, _ = extract_pdf(ADVERSARIAL.read_bytes(), max_pages=10, max_chars_per_page=20_000)
    chunks = chunk_pages("f" * 64, pages, size=1200, overlap=200)
    assert any("possible_injection" in c.flags for c in chunks)


# ---------------------------------------------------------------- pipeline
def test_duplicate_ingestion_is_idempotent(settings):
    store, emb = MemoryStore("hashing-1024"), HashingEmbedder(settings.embedding_dim)
    data = POLICIES[0].read_bytes()
    first = ingest_bytes(data, "a.pdf", settings=settings, store=store, embedder=emb)
    second = ingest_bytes(data, "renamed-copy.pdf", settings=settings, store=store, embedder=emb)
    assert first.status == "ingested" and second.status == "duplicate"
    assert second.doc_id == first.doc_id
    assert len(store.list_documents()) == 1
    assert store.list_documents()[0].num_chunks == first.num_chunks


def test_ingest_rejects_oversized_upload(settings):
    s = settings.model_copy(update={"max_upload_mb": 0})
    with pytest.raises(PdfValidationError, match="exceeds"):
        ingest_bytes(POLICIES[0].read_bytes(), "a.pdf", settings=s, store=MemoryStore("x"),
                     embedder=HashingEmbedder(64))


def test_failed_ingestion_leaves_store_untouched(settings):
    store = MemoryStore("hashing-1024")
    with pytest.raises(PdfValidationError):
        ingest_bytes(b"%PDF-1.4 garbage", "bad.pdf", settings=settings, store=store, embedder=HashingEmbedder(64))
    assert store.list_documents() == []


def test_reset_db_on_memory_store_fails_cleanly(monkeypatch, capsys):
    # Audit L1: used to crash with AttributeError.
    from policyqa import cli
    from policyqa.config import get_settings

    monkeypatch.setenv("VECTOR_STORE", "memory")
    get_settings.cache_clear()
    try:
        assert cli.main(["reset-db", "--yes"]) == 2
        assert "only applies to VECTOR_STORE=postgres" in capsys.readouterr().err
    finally:
        get_settings.cache_clear()
