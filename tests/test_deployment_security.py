"""Static regression checks for local-only deployment boundaries."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def read_repository_file(relative_path: str) -> str:
    """Return a UTF-8 repository file's contents.

    Args:
        relative_path: Path relative to the repository root.

    Returns:
        The file contents.
    """
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


class DeploymentSecurityTests(unittest.TestCase):
    """Protect the local-only and secret-free deployment configuration."""

    def test_launchers_bind_vite_to_loopback_explicitly(self) -> None:
        """Both launchers must pass an address after Vite's ``--host`` flag."""
        bash_launcher = read_repository_file("start_all.sh")
        powershell_launcher = read_repository_file("windows/start_all.ps1")

        self.assertIn(
            'run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"',
            bash_launcher,
        )
        self.assertIn(
            "'--host', '127.0.0.1', '--port', \"$FrontendPort\"",
            powershell_launcher,
        )

    def test_launchers_do_not_embed_the_historical_tiled_credential(self) -> None:
        """Legacy-key rotation must compare fingerprints, not retain the secret."""
        bash_launcher = read_repository_file("start_all.sh")
        powershell_launcher = read_repository_file("windows/start_all.ps1")

        self.assertNotIn("LEAKED_TILED_KEY=", bash_launcher)
        self.assertNotIn("$leaked =", powershell_launcher)
        self.assertIn("LEGACY_TILED_KEY_SHA256", bash_launcher)
        self.assertIn("$legacyKeySha256", powershell_launcher)

    def test_compose_publishes_the_api_on_loopback_only(self) -> None:
        """Docker Compose must not publish the unauthenticated API on every NIC."""
        compose = read_repository_file("docker-compose.yml")

        self.assertIn('"127.0.0.1:8002:8002"', compose)
        self.assertNotIn('- "8002:8002"', compose)

    def test_docker_context_excludes_secrets_and_local_models(self) -> None:
        """Generated credentials and mutable model snapshots must stay out of builds."""
        ignore_lines = {
            line.strip() for line in read_repository_file(".dockerignore").splitlines()
        }

        self.assertIn("**/.env", ignore_lines)
        self.assertIn("**/.env.*", ignore_lines)
        self.assertIn("!**/.env.example", ignore_lines)
        self.assertIn("frontend/public/models/", ignore_lines)

    def test_runtime_image_is_non_root_and_health_checked(self) -> None:
        """The final image must drop privileges and expose container health."""
        dockerfile = read_repository_file("Dockerfile")

        self.assertIn("USER app", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)

    def test_frontend_declares_node_twenty_or_newer(self) -> None:
        """The package metadata and version file must match the supported runtime."""
        package = json.loads(read_repository_file("frontend/package.json"))
        node_version = read_repository_file(".nvmrc").strip()

        self.assertEqual(package["engines"]["node"], ">=20")
        self.assertEqual(node_version, "20")

    def test_documentation_no_longer_advertises_node_eighteen(self) -> None:
        """User-facing setup docs must not direct users to an unsupported Node version."""
        document_paths = [
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "windows/README.md",
            *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
        ]

        for document_path in document_paths:
            document = document_path.read_text(encoding="utf-8")
            self.assertNotIn("Node.js 18+", document)
            self.assertNotIn("18 or newer", document)

    def test_tiled_config_contains_no_personal_absolute_paths(self) -> None:
        """A fresh clone must not depend on a developer's home directory."""
        tiled_config = read_repository_file("tiled/config.yml")

        self.assertNotIn("/Users/", tiled_config)
        self.assertNotIn("C:\\Users\\", tiled_config)


if __name__ == "__main__":
    unittest.main()
