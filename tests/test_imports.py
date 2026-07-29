import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_classifier_import_does_not_pull_langgraph():
    """from audit.classifier import identify 不应触发 langgraph 导入（轻量、可独立用）。"""
    code = (
        "import audit.classifier\n"
        "import sys\n"
        "langgraph_mods = sorted(m for m in sys.modules if 'langgraph' in m)\n"
        "assert not langgraph_mods, langgraph_mods\n"
        "from audit.classifier import identify\n"
        "assert identify('你好')['risk_type'] == 'safe'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
