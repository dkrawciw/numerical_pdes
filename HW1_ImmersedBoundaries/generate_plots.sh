uv sync
source .venv/bin/activate

echo "Generating Plots..."

uv run problem1_1.py &
uv run problem1_2.py &

deactivate