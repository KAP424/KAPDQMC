"""Entry point for the independent Ref-calibration audit.

python src/Measure/ai/audit_ref_calibration.py --stage=final

Implementation lives with its audit artifacts; it never imports the primary
analysis script or its fit function.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent /
                      "analysis_ref_calibration/audit_calibration.py"), run_name="__main__")
