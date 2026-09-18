# Formula and theorem audit ledger

Populate one row per equation or theorem after screening the cited full text.

| ID | Code location | Formula/theorem | Type | Assumptions | Derivation/proof source | Domain/units | Numerical check | Contradiction | Status |
|---|---|---|---|---|---|---|---|---|---|
| F-001 | `vaccinegpt/five_layer.py` | `v0 = f(tau_TI)` | derived model coupling | bounded tau, detached upstream state | cite included source or mark hypothesis | dimensionless/declared units | `verification/contract.py` | search included evidence | pending |
| F-002 | `vaccinegpt/losses.py` | non-negative PU risk | cited/derived loss | class-prior assumptions | cite original method | scalar risk | `verification/verify_math.py` | compare competing PU results | pending |
| F-003 | `vaccinegpt/l0_clef.py` | VIB KL | cited objective | Gaussian posterior/prior | cite source | scalar objective | finite-gradient test | report conflicts | pending |

No row may be marked proven solely because the implementation runs.
