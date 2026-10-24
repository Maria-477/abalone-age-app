"""
Adds (or replaces) the link cell at the very top of the notebook and fills the links into README.md.

Run it ONLY after deployment succeeded, with your real URLs:

    python update_notebook_links.py https://YOUR-APP.streamlit.app https://github.com/YOUR-USER/YOUR-REPO
"""
import json
import sys

NOTEBOOK = "abalone_age_prediction_10fold.ipynb"
TITLE = "Abalone Age Prediction System – 10-Fold Cross Validation"
MARKER = "<!-- deployment-links -->"

if len(sys.argv) != 3 or not all(u.startswith("https://") for u in sys.argv[1:]):
    sys.exit("Usage: python update_notebook_links.py <live-streamlit-url> <github-repo-url>")
live_url, repo_url = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

text = (
    f"{MARKER}\n"
    f"**Live Streamlit Application:** [{live_url}]({live_url})\n\n"
    f"**GitHub Repository:** [{repo_url}]({repo_url})\n\n"
    f"**Project Title:** {TITLE}\n"
)
cell = {"cell_type": "markdown", "id": "deploylinks", "metadata": {}, "source": text.splitlines(keepends=True)}

if nb["cells"] and MARKER in "".join(nb["cells"][0]["source"]):
    nb["cells"][0] = cell
else:
    nb["cells"].insert(0, cell)

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated: links added at the very top.")

with open("README.md", encoding="utf-8") as f:
    readme = f.read()
readme = readme.replace("{{LIVE_APP_URL}}", live_url).replace("{{GITHUB_REPO_URL}}", repo_url)
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("README updated with both links.")
