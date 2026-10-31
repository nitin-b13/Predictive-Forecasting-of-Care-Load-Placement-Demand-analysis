"""Run the full pipeline: python run_pipeline.py [--skip-backtest]
The walk-forward backtest takes roughly 25-30 minutes on a single CPU; pass --skip-backtest to reuse
outputs/backtest_forecasts.csv and only refresh metrics, forecasts and figures."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
steps = ["prepare.py", "backtest.py", "evaluate.py", "finalize.py", "figures.py", "dashboard_html.py"]
if "--skip-backtest" in sys.argv:
    steps.remove("backtest.py")
for s in steps:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipeline" / s)], check=True)
print("\nDone. Launch the dashboard with: streamlit run app.py")
