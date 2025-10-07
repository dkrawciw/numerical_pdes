uv sync
source .venv/bin/activate

echo "Generating Plots..."

uv run problem1_1.py &
uv run problem1_2.py &
uv run problem2_1.py &
uv run problem2_2.py &
uv run problem3_1.py &
uv run problem3_2.py &
uv run problem4_1.py &
uv run problem4_2.py &

deactivate