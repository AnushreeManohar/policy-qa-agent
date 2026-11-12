"""Command-line entry point: `policyqa init-db | reset-db | ingest | ask | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policyqa.config import get_settings
from policyqa.embeddings import build_embedder
from policyqa.ingestion.pdf import PdfValidationError
from policyqa.ingestion.pipeline import ingest_bytes
from policyqa.logging_setup import configure_logging
from policyqa.store import StoreError, build_store


def _components():  # type: ignore[no-untyped-def]
    settings = get_settings()
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.name)
    return settings, embedder, store


def cmd_init_db(_: argparse.Namespace) -> int:
    _, _, store = _components()
    store.init_schema()
    print("schema ready")
    return 0


def cmd_reset_db(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to drop tables without --yes", file=sys.stderr)
        return 2
    _, _, store = _components()
    if not hasattr(store, "reset"):
        print("reset-db only applies to VECTOR_STORE=postgres (the memory store is empty on every start)",
              file=sys.stderr)
        return 2
    store.reset()
    store.init_schema()
    print("database reset")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings, embedder, store = _components()
    store.init_schema()
    paths: list[Path] = []
    for p in map(Path, args.paths):
        paths.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    failures = 0
    for path in paths:
        try:
            r = ingest_bytes(path.read_bytes(), path.name, settings=settings, store=store, embedder=embedder,
                             policy_type=args.policy_type)
            print(f"{r.status:9} {r.filename}  pages={r.num_pages} chunks={r.num_chunks} "
                  f"flagged={r.flagged_chunks} type={r.policy_type}")
        except (PdfValidationError, OSError) as exc:
            failures += 1
            print(f"FAILED    {path.name}: {exc}", file=sys.stderr)
    store.close()
    return 1 if failures else 0


def cmd_ask(args: argparse.Namespace) -> int:
    from policyqa.agent import PolicyQAAgent
    from policyqa.retrieval import HybridRetriever

    settings, embedder, store = _components()
    agent = PolicyQAAgent(HybridRetriever(store, embedder, settings), settings)
    result = agent.ask(args.question, args.policy_type)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"[{result.status} | {result.mode} | {result.latency_ms:.0f} ms]\n\n{result.answer}\n")
        for c in result.citations:
            print(f"  [{c.source_id}] {c.title} — {c.section or 'n/a'}, p.{c.page_start}")
    store.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("policyqa.api.app:app_factory", factory=True, host=args.host, port=args.port, log_config=None)
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_logging(get_settings().log_level)
    ap = argparse.ArgumentParser(prog="policyqa")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db").set_defaults(fn=cmd_init_db)
    r = sub.add_parser("reset-db")
    r.add_argument("--yes", action="store_true")
    r.set_defaults(fn=cmd_reset_db)
    i = sub.add_parser("ingest")
    i.add_argument("paths", nargs="+")
    i.add_argument("--policy-type")
    i.set_defaults(fn=cmd_ingest)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--policy-type")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container entrypoint
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
