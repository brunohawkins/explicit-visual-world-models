# Explicit visual world models

## Two stages (plain English)

| Name | What happens | What you need |
|------|----------------|---------------|
| **P1 — write the simulator** | The model inspects demo clips and writes `simulator_gen.py` (state, physics, goal). A gate checks that the code is consistent. | Gemini API key. Slow (tens of minutes). |
| **P2 — use the simulator** | A planner (CEM + MPC) rolls the written code forward and applies the best actions in the task environment. | No API key if you use the frozen example below. |

The thesis reported 80 P1 attempts and 50 control cases per simulator. The demos here are smaller, so it is quicker to run and check.

Re-running P1 will **not** match the thesis numbers bit-for-bit: Gemini is a hosted model with no dated snapshot.

---

## Contents

```text
README.md                 
release_contract.json     frozen campaign pins (model, hashes, 80-run matrix)
pyproject.toml            Python package (install this)

prompts/action_conditioned/   the one prompt used for all final runs
src/vdaworld/                 generation + planning code
scripts/demo_p1.py            one-command P1 demo
scripts/demo_p2.py            one-command P2 demo
scripts/smoke_p1.py           full P1 entry point (same as the campaign)
scripts/smoke_p2_mpc.py       full P2 entry point (same as the campaign)
tests/                        unit tests (no API key)

dataset/                  RGB–action demos used in the thesis (~95 MB)
  two_room/               red agent, two rooms, a doorway
  reacher/                2-joint arm
  pusht/                  pusher + T-shaped block
  cube/                   pick-and-place cube
  each of those has:
    expert_only/          eight good demonstrations (the default demo cell)
    expert_noisy/ …       other mixture cells from the composition study
    _sandbox/             the two frames the generator is allowed to “fit”
    _audit/               a held-out clip the generator never sees

manifests/                the 50 start/goal cases used for P2 (seed 42)

examples/two_room_expert_only_r1/
  simulator_gen.py        a simulator the campaign actually generated
  runtime_contract.json   how it was allowed to call tools at test time
  metrics.json            P1 gate result from the campaign
```

**Not included:** API keys, the 80 full campaign logs and videos, cluster job scripts, and the latent baseline (LeWM) training code. Too many GB to push to github.

---

## Install

Python **3.13**. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

If you use `uv`:

```bash
uv sync
```

Optional check (no Gemini, no GPU):

```bash
python -m pytest tests/test_cem.py tests/test_mpc.py tests/test_interfaces.py -q
```

---

## Run the P2 demo first (no API key)

This loads the frozen Two-Room simulator, plans for **one** start/goal pair, and writes a video.

```bash
python scripts/demo_p2.py
```

Look in `viz_output/demo_p2/demo_p2/` for `summary.json` and `seed_42/mpc.mp4`.

Closer to the thesis planner (still only 5 cases, not 50; slower):

```bash
python scripts/demo_p2.py --faithful
```

This uses a **built-in Two-Room environment**, not the isolated LeWM/SWM server the thesis used for the published 50-case table. Same start/goal list; enough to see that planning runs.

---

## Optional: run the P1 demo (needs a Gemini key)

```bash
export GEMINI_API_KEY=your_key_here
python scripts/demo_p1.py
```

This is **one** Two-Room generation on the eight `expert_only` clips, with the same gate flags as the campaign. Output: `viz_output/demo_p1/simulator_gen.py`.

You can then point P2 at that new file:

```bash
python scripts/demo_p2.py --sim-path viz_output/demo_p1/simulator_gen.py
```

---