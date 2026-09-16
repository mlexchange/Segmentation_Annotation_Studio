"""CLI entrypoint ``ipred`` with --project session sugar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ipred import feature_setups, preprocess, train_infer
from ipred.catalog import Catalog
from ipred.trainers import list_trainers


def _catalog() -> Catalog:
    cat = Catalog()
    feature_setups.ensure_default_setups(cat)
    return cat


def _resolve_session(args: argparse.Namespace, cat: Catalog) -> str:
    """Return session_id from --session or --project sugar."""
    if getattr(args, "session", None):
        return str(args.session)
    if getattr(args, "kind", None) and getattr(args, "source", None):
        session = cat.open_session(
            kind=args.kind,
            source=args.source,
            server_uri=getattr(args, "server_uri", None),
            root=getattr(args, "root", None),
        )
        print(json.dumps({"session_id": session.session_id, "project_id": session.project_id}))
        return session.session_id
    raise SystemExit("provide --session or --kind/--source (--project sugar)")


def _add_project_sugar(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", default=None, help="Existing session id")
    parser.add_argument("--kind", default=None, help="Project kind (local|tiled)")
    parser.add_argument("--source", default=None, help="Project source path")
    parser.add_argument("--server-uri", dest="server_uri", default=None)
    parser.add_argument("--root", default=None, help="Local root override")


def main(argv: list[str] | None = None) -> None:
    """CLI main."""
    parser = argparse.ArgumentParser(
        prog="ipred",
        description="Iterative prediction backend CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_session = sub.add_parser("session", help="Session commands")
    sess_sub = p_session.add_subparsers(dest="session_cmd", required=True)
    p_open = sess_sub.add_parser("open", help="Open a session for a project")
    p_open.add_argument("--kind", required=True)
    p_open.add_argument("--source", required=True)
    p_open.add_argument("--server-uri", dest="server_uri", default=None)
    p_open.add_argument("--root", default=None)

    p_setup = sub.add_parser("setup", help="Feature Setup commands")
    setup_sub = p_setup.add_subparsers(dest="setup_cmd", required=True)
    setup_sub.add_parser("list", help="List setups")
    p_show = setup_sub.add_parser("show", help="Show one setup")
    p_show.add_argument("setup_id")

    p_pre = sub.add_parser("preprocess", help="Featurize (cache-aware)")
    _add_project_sugar(p_pre)
    p_pre.add_argument("--setup", required=True, help="Feature setup id")
    p_pre.add_argument("--slice", type=int, default=0)

    p_train = sub.add_parser("train", help="Train a model")
    _add_project_sugar(p_train)
    p_train.add_argument("--labels", required=True, help="JSON file of shapes")
    p_train.add_argument("--trainer", default="catboost")
    p_train.add_argument("--feature-id", dest="feature_id", default=None)
    p_train.add_argument("--config", default=None, help="JSON config file or string")

    p_inf = sub.add_parser("infer", help="Infer + conformal maps")
    _add_project_sugar(p_inf)
    p_inf.add_argument("--alpha", type=float, default=0.05)
    p_inf.add_argument("--model-id", dest="model_id", default=None)
    p_inf.add_argument("--feature-id", dest="feature_id", default=None)

    p_rt = sub.add_parser("rethreshold", help="Rethreshold from cached proba")
    _add_project_sugar(p_rt)
    p_rt.add_argument("--alpha", type=float, required=True)
    p_rt.add_argument("--run-id", dest="run_id", default=None)

    sub.add_parser("trainers", help="List trainer plugins")

    args = parser.parse_args(argv)
    cat = _catalog()

    if args.cmd == "session" and args.session_cmd == "open":
        session = cat.open_session(
            kind=args.kind,
            source=args.source,
            server_uri=args.server_uri,
            root=args.root,
        )
        _print(
            {
                "session_id": session.session_id,
                "project_id": session.project_id,
            }
        )
        return

    if args.cmd == "setup":
        if args.setup_cmd == "list":
            _print({"setups": feature_setups.list_setups()})
            return
        if args.setup_cmd == "show":
            _print(feature_setups.resolve_setup(args.setup_id))
            return

    if args.cmd == "trainers":
        _print({"trainers": list_trainers()})
        return

    if args.cmd == "preprocess":
        sid = _resolve_session(args, cat)
        out = preprocess.run_preprocess(
            cat,
            session_id=sid,
            feature_setup_id=args.setup,
            slice_index=args.slice,
        )
        _print(out)
        return

    if args.cmd == "train":
        sid = _resolve_session(args, cat)
        shapes = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        if isinstance(shapes, dict) and "shapes" in shapes:
            shapes = shapes["shapes"]
        config: dict[str, Any] = {}
        if args.config:
            cfg_path = Path(args.config)
            if cfg_path.is_file():
                config = json.loads(cfg_path.read_text(encoding="utf-8"))
            else:
                config = json.loads(args.config)
        out = train_infer.run_train(
            cat,
            session_id=sid,
            shapes=shapes,
            feature_id=args.feature_id,
            trainer_id=args.trainer,
            config=config,
        )
        _print(out)
        return

    if args.cmd == "infer":
        sid = _resolve_session(args, cat)
        out = train_infer.run_infer(
            cat,
            session_id=sid,
            model_id=args.model_id,
            feature_id=args.feature_id,
            alpha=args.alpha,
        )
        _print(out)
        return

    if args.cmd == "rethreshold":
        sid = _resolve_session(args, cat)
        out = train_infer.run_rethreshold(
            cat,
            session_id=sid,
            run_id=args.run_id,
            alpha=args.alpha,
        )
        _print(out)
        return

    parser.error(f"unhandled command {args.cmd}")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
