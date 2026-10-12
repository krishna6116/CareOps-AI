"""Export the static GitHub Pages demo from the committed pipeline run.

Calls the real FastAPI app in-process (against a throwaway SQLite database)
for every GET path the dashboard reads, plus the assistant's example
questions and the simulator presets, and writes the responses to
``frontend/public/demo``. The dashboard built with
``VITE_DEMO_MODE=snapshot`` reads these files instead of a live API.

    python deploy/export_demo_snapshot.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "frontend" / "src" / "demo-config.json"
OUTPUT = ROOT / "frontend" / "public" / "demo"


def slug(path: str) -> str:
    """Path to file name; must match ``snapshotFile`` in frontend/src/api.ts."""
    return re.sub(r"[^A-Za-z0-9]+", "_", path.lstrip("/")).rstrip("_") + ".json"


def scenario_key(admissions_multiplier: float, icu_bed_delta: int, staff_delta: float) -> str:
    """Must match ``scenarioKey`` in frontend/src/api.ts."""
    return f"a{admissions_multiplier:.2f}_i{int(icu_bed_delta)}_s{staff_delta:.2f}"


def main() -> None:
    database = Path(tempfile.mkdtemp()) / "demo.db"
    os.environ["CAREOPS_DATABASE_URL"] = f"sqlite:///{database.as_posix()}"
    os.environ["CAREOPS_LLM_PROVIDER"] = "deterministic"
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from backend.app.main import app
    from backend.app.security import RateLimiter
    import backend.app.security as security

    security.interactive_limiter = RateLimiter(limit=10_000)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("*.json"):
        stale.unlink()

    with TestClient(app) as client:
        simulations = {}
        for preset in config["scenarioPresets"]:
            body = {
                "scenario_name": f"demo_{preset['label']}",
                "admissions_multiplier": 1 + preset["admissionsPercent"] / 100,
                "icu_bed_delta": preset["icuBedDelta"],
                "staff_availability_delta": preset["staffPercent"] / 100,
            }
            response = client.post("/simulation/run", json=body)
            response.raise_for_status()
            key = scenario_key(body["admissions_multiplier"], body["icu_bed_delta"],
                               body["staff_availability_delta"])
            simulations[key] = response.json()

        answers = {}
        for question in config["exampleQuestions"]:
            response = client.post("/assistant/ask", json={"question": question})
            response.raise_for_status()
            answers[question] = response.json()

        # GET paths last, so simulation and assistant history include the runs above.
        files = {}
        for path in config["snapshotPaths"]:
            response = client.get(path)
            response.raise_for_status()
            name = slug(path)
            (OUTPUT / name).write_text(json.dumps(response.json()), encoding="utf-8")
            files[path] = name

    (OUTPUT / "simulations.json").write_text(json.dumps(simulations), encoding="utf-8")
    (OUTPUT / "assistant.json").write_text(json.dumps(answers), encoding="utf-8")
    manifest = json.loads(
        (ROOT / "optimization" / "resource_planning" / "outputs" / "portfolio_run_manifest.json")
        .read_text(encoding="utf-8")
    )
    (OUTPUT / "snapshot.json").write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenario_as_of_date": manifest.get("scenario_as_of_date"),
        "files": files,
        "assistant_questions": list(answers),
        "simulation_presets": list(simulations),
    }, indent=2), encoding="utf-8")
    total = sum(path.stat().st_size for path in OUTPUT.glob("*.json"))
    print(f"Wrote {len(files) + 3} files ({total / 1e6:.1f} MB) to {OUTPUT}")


if __name__ == "__main__":
    main()
