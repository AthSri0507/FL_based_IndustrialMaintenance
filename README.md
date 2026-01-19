# Federated Predictive Maintenance

Minimal scaffold for federated RUL prediction and fault detection.

Structure:
- src/: source code
- data/: datasets
- notebooks/: experiments and analysis
- configs/: configuration files
- experiments/outputs/: saved experiment artifacts and snapshots

Quick start
1. Create a virtual environment and install dependencies

	 - Windows (PowerShell):

		 ```powershell
		 python -m venv .venv
		 .\.venv\Scripts\Activate.ps1
		 pip install -r requirements.txt
		 ```

	 - macOS / Linux:

		 ```bash
		 python -m venv .venv
		 source .venv/bin/activate
		 pip install -r requirements.txt
		 ```

2. Recommended environment variables

	 - `PYTHONHASHSEED=42` — improves reproducibility (the project sets this when using the reproducibility utilities).
	 - You can place env vars in a `.env` file or export them in your shell before running.

3. Run the server (development)

	 - Module entry (project-provided):

		 ```powershell
		 python -m src.server.main
		 ```

	 - Or use Uvicorn for the FastAPI app:

		 ```bash
		 uvicorn src.server.app:app --reload --host 0.0.0.0 --port 8000
		 ```

4. Run experiments and notebooks

	 - Start Jupyter Lab / Notebook and open the files in `notebooks/` (e.g., `03_data_profile_comparison.ipynb`, `11_comprehensive_fl_comparison.ipynb`).

		 ```bash
		 jupyter lab
		 ```

5. Outputs

	 - Experiment outputs, logs, and exported snapshots are written to `experiments/outputs/`.

Notes
- The project includes `src/utils/reproducibility.py` which helps set seeds and capture environment info.
- See `src/server` for the simple FastAPI-based FL server scaffold and `src/clients` for client-side code.
- For a quick comparison workflow, check `notebooks/11_comprehensive_fl_comparison.ipynb` which loads saved experiment outputs only.
