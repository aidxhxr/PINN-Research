# Pre-registered predictions — prospective test of the reachability table

**Written 2026-08-02, BEFORE any fit in `runs/20260802_protocol_test/` was
run.** The protocols and the anchor ratios below come from `anchor_reach.py`,
which trains nothing and needs only reference trajectories, so they were fixed
in advance. The error columns are what the experiment measures.

## What is being tested

`anchor_reach.py` produces a design table: for every edge, the cheapest
knockdown protocol that puts that edge's `f(0) = 0` anchor into the data. It is
computed without training anything. If the anchor account is right, that table
is a **prediction sheet** — it should say in advance which edges become
learnable and under what experiment.

The two edges chosen are the **worst production edges in the whole atlas**, and
neither has ever been run under any depletion condition. Their 10-condition
numbers (`runs/20260801_204046_screen_atlas`, gated, Normal; Severe from the
negative-control run):

| edge | basal | Normal fNRMSE / basal | Severe fNRMSE / basal | eq params |
|---|---|---|---|---|
| `m_h13` (MYC → HOXA13) | `a13` | 57.2% / 12.8% | 11.6% / 95.6% | 2/8 |
| `h13_b` (HOXA13 → β-cat) | `W` | 19.3% / 16.2% | 171.4% / 57.5% | — |

`h13_b` in Severe at 171.4% is the single worst functional result in the repo.

## Design — each edge gets a near-miss control

Three protocols per edge. The middle one is the point: it is a **larger**
perturbation than doing nothing, but it does **not** reach that edge's anchor.
If the anchor is what matters, it should behave like the baseline.

**`m_h13`** — regulator `m`, basal `a13`:

| protocol | what it is | `m` ratio (worst of 4 regimes) | reaches? |
|---|---|---|---|
| `none` | the existing 10 conditions | 0.1155 | no |
| `bcatKO` | `kW = 0`, `k13 = 0` — a double knockout of β-catenin production | 0.0786 | **no** |
| `mycKO` | `bcatKO` + `kaM = 0` | **0.0000** | **yes** |

**`h13_b`** — regulator `h13`, basal `W`:

| protocol | what it is | `h13` ratio | reaches? |
|---|---|---|---|
| `none` | the existing 10 conditions | 0.1926 | no |
| `mycKO` | `kW = 0`, `k13 = 0`, `kaM = 0` | 0.1191 | **no** |
| `hox13KO` | `mycKO` + `ka13 = 0` | **0.0000** | **yes** |

## Predictions

**Q1 — the prescribed protocol fixes the basal parameter.** Under the reaching
protocol, `m_h13`'s `a13` error falls below **3%** in both regimes, from
12.8% (Normal) and 95.6% (Severe).

**Q2 — the near-miss protocol does not.** Under `bcatKO`, `m_h13`'s `a13` error
stays above 8% in at least one regime and shows no systematic improvement over
`none`, despite `bcatKO` being a double genetic knockout — a far larger
perturbation than any of the ten base conditions. This is the falsifier: if
`bcatKO` fixes `m_h13` too, then perturbation size is doing the work and the
anchor table is not predictive.

**Q3 — equation-parameter count moves only for the reaching protocol.**
`m_h13`'s recovered count rises under `mycKO` and does not under `bcatKO`. This
is the more robust statistic; the 2026-08-01 negative control turned on exactly
this quantity failing to move (2/8 → 2/8).

**Q4 — functional error improves alongside.** `m_h13`'s functional NRMSE falls
below 20% in Normal under `mycKO`, from 57.2%. Weaker than Q1–Q3 because
functional error at this edge has been unstable across restarts.

**Q5 — `h13_b`, with an acknowledged confound.** Under `hox13KO` the basal `W`
error falls below 5% in both regimes. **This cell is secondary and cannot carry
the claim**, for a reason recorded before the run: `h13_b`'s basal parameter
*is* `W`, and every protocol that reaches `h13`'s anchor necessarily sets
`kW = 0` — because `h13` cannot be driven to zero without first driving `b` to
zero. So the reaching arm also removes `W` from that condition entirely.

Note this is *not* the same confound as the 2026-08-01 negative control, where
`kW = 0.02` meant two conditions observed `W` and `0.02·W` and thereby
identified `W` outright. Setting `kW = 0` exactly observes `0·W`, which carries
**no** information about `W`. The confound here is different and milder, but it
is still a confound, and `h13_b` is reported as supporting evidence only.
`m_h13` is the cell that decides Q1–Q4.

## What would sink the design table

* **Q2 fails** — the near-miss double knockout fixes `m_h13` anyway. Then the
  table is measuring perturbation magnitude, not anchor visitation.
* **Q1 and Q3 both fail** — the prescribed protocol does not help. Then the
  anchor account does not generalise past the four edges it was built on, and
  the reachability table should not be published as a design tool.

## Cost caveat recorded in advance

`mycKO` and `hox13KO` require silencing a *basal production* term (`kaM`,
`ka13`), which in a real lab means editing a constitutive promoter — materially
harder than the siRNA-scale knockdowns (`kW`, retinoid-free medium) that fix
the RA and β-catenin edges. If these predictions hold, the honest framing is
"reachable, but expensive", not "solved".
