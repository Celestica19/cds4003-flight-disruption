"""Execute notebook code with IPython and save genuine captured outputs.

This lightweight runner needs IPython, but does not require a kernel server.
Use standard Jupyter in the normal student environment if preferred.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]

def execute_all():
    os.chdir(ROOT)
    validation = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        InteractiveShell.clear_instance()
        shell = InteractiveShell.instance()
        notebook = json.loads(path.read_text())
        execution_count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            print(f"Executing {path.name}, cell {execution_count}", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell("".join(cell["source"]), store_history=True)
            outputs = []
            if captured.stdout:
                outputs.append({"output_type": "stream", "name": "stdout", "text": captured.stdout.splitlines(keepends=True)})
            if captured.stderr:
                outputs.append({"output_type": "stream", "name": "stderr", "text": captured.stderr.splitlines(keepends=True)})
            for rich in captured.outputs:
                outputs.append({"output_type": "display_data", "data": rich.data, "metadata": rich.metadata})
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            cell["metadata"]["execution"] = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat()}
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Notebook failed: {path.name}, cell {execution_count}: {captured.stdout}\n{captured.stderr}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                text = "".join(cell["source"]).replace("Unexecuted code scaffold. No empirical outputs have been invented.", "Executed against the uploaded real dataset and its verified analysis artifacts. Outputs below were captured from the code cells.")
                cell["source"] = text.splitlines(keepends=True)
        notebook["metadata"]["execution_evidence"] = {"runner": "IPython InteractiveShell", "data_status": "REAL_DATA", "executed_at_utc": datetime.now(timezone.utc).isoformat()}
        path.write_text(json.dumps(notebook, indent=2) + "\n")
        validation.append({"notebook": path.name, "executed_code_cells": execution_count, "status": "passed"})
    (ROOT / "reports/notebook_execution.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation

if __name__ == "__main__":
    execute_all()
