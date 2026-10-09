# vsa_cleanup — VSA attribute query with codebook cleanup

**Category**: vsa · **Semiring(s)**: real (+,×) · **Execution structure**: dag
**Module**: `softwareworkload/vsa_cleanup/workload.py` · **Test**: `tests/test_workload_vsa_cleanup.py`

## 1. Problem and reference algorithm
Four attributes with 16 symbols each; each symbol is a Gaussian hypervector of dimension d.
Attribute keys and object-slot keys are bipolar (±1), so a key is its own inverse under the
elementwise product. An object is the bundle of its four (key ⊙ symbol) bindings, a scene is the
bundle of its J = 3 objects each bound to a slot key, and a query (scene, slot, attribute)
unbinds the scene with slot ⊙ key and cleans up against the 16 symbols of that attribute:

    E(o)   = Σ_i key_i ⊙ sym(i, v(o,i))          SC(s) = Σ_j slot_j ⊙ E(o(s,j))
    U(q)   = SC(s_q) ⊙ slot_{j_q} ⊙ key_{i_q}    sim(q,k) = ⟨U(q), sym(i_q, k)⟩    ans(q) = argmax_k

`native()` is numpy: elementwise products and sums for the encodings, `CB[i] @ u` per query, and
argmax with first-index ties. It counts the multiply-accumulates of the similarity search
(Q × 16 × d) plus every elementwise bind (objects O·4·d, scenes S·J·d, composite keys J·4·d,
unbind Q·d). Scales: small d=2,048, Q=32 queries over S=8 scenes; large d=8,192, Q=128 over S=11
scenes (J=3 slots, 4 attributes, 16 symbols at both). Ground truth is the encoded symbol; the
answer of both sides is compared exactly and the 16 similarities within tolerance.

## 2. Mathematical decomposition
Everything is in (+,×) over the reals with d as the join attribute:

    E(o,d)   = Σ_{i,k} OBJ(o,i,k) · CB(i,k,d) · KEY(i,d)            bind on d, bundle over i
    SC(s,d)  = Σ_{j,o} SCN(s,j,o) · E(o,d) · SLOT(j,d)              bind on d, bundle over j
    U(q,i,d) = Σ_{s,j} QRY(q,s,j,i) · SC(s,d) · SLOT(j,d) · KEY(i,d)  unbind (single term)
    SIM(q,k) = Σ_d U(q,i,d) · CB(i,k,d)                              the cleanup GEMM
    ans(q)   = argmax_k SIM(q,k)                                     select per q
OBJ, SCN and QRY are 1.0-valued indicator relations (which symbol an object has, which object
sits in a slot, which (scene, slot, attribute) a query asks for). No iteration: a DAG of nine
operators plus Q selects.

## 3. Algebra program
    G   = combine(OBJ(o,i,k), CB(i,k,d))               -> (o,i,k,d)   gather symbol vectors
    E   = combine(G, KEY(i,d), keep=(o,d))              -> (o,d)       bind with key_i, Σ_i
    G2  = combine(SCN(s,j,o), E(o,d))                   -> (s,j,o,d)   gather objects per scene
    SC  = combine(G2, SLOT(j,d), keep=(s,d))            -> (s,d)       bind with slot_j, Σ_j
    QK  = combine(SLOT(j,d), KEY(i,d))                  -> (j,d,i)     composite unbind keys
    G3  = combine(QRY(q,s,j,i), SC(s,d))                -> (q,s,j,i,d) gather the scene per query
    U   = combine(G3, QK(j,d,i), keep=(q,i,d))          -> (q,i,d)     unbind
    SIM = combine(U(q,i,d), CB(i,k,d), keep=(q,k))      -> (q,k)       GEMM: Q·16·d MACs
    for each q: ans(q) = select_k(filter(SIM, key[0] == q), 1)

## 4. Hotspot event and work efficiency
Event: one multiply-accumulate of the similarity search or one elementwise bind. The GEMM
combine performs exactly Q·16·d ⊗ (`gemm_mul_ops == native_macs` is asserted by the test).

Measured (harness, 2026-09-25; counts are size-determined, identical for seeds 0–2):

| scale | native_ops = MACs + binds | algebra ⊗ | ratio_mul | ratio_sr | ratio_cand | select-phase candidates | accuracy (golden / algebra) | t_native / t_algebra (s) |
|---|---|---|---|---|---|---|---|---|
| small (seed 0) | 1,384,448 = 1,048,576 + 335,872 | 1,695,744 | 1.225 | 2.112 | 1.237 | 16,896 (1.0% of candidates) | 1.00 / 1.00 | 0.001 / 2.10 |
| large (seed 0) | 19,275,776 = 16,777,216 + 2,498,560 | 21,676,032 | 1.125 | 2.046 | 1.138 | 264,192 (1.2% of candidates) | 1.00 / 1.00 | 0.004 / 22.16 |

ratio_mul > 1 comes entirely from the gathers: each indicator join (G, G2, G3) is counted as one
⊗ per gathered element, so the object, scene and unbind phases cost twice their bind count
(small seed 0, ⊗ per phase: encode_objects 393,216 for 196,608 binds,
encode_scenes 98,304 for 49,152, unbind 131,072 for 65,536,
query_keys 24,576 for 24,576; GEMM 1,048,576 for 1,048,576 MACs). The
GEMM is 61.8% of the algebra's ⊗ at small and 77.4% at large, which is why the
ratio falls with Q and d. ratio_sr ≈ 2 because a MAC is one ⊗ and one ⊕ and the native count is
MACs (GEMM ⊕: 1,048,064 at small). ratio_cand ≈ ratio_mul + the select phase.
U is fully dense (nnz / (Q·d) = 1.00): Gaussian symbols leave no exact zeros for the
relation layer to suppress, so the MAC count is not reduced by value sparsity.

## 5. scan() functions used
none (binds are joins on d; the per-query restriction is a filter).

## 6. Control residue
None: the Python loop over the Q queries only issues the filter and select operators; every
arithmetic and every comparison is inside an operator. Accuracy against the ground-truth
encoding is 1.0 on both sides at both scales (d ≫ the 12 bundled terms of a scene).

## 7. Where the algebra is insufficient or awkward
1. **No INDEX / gather.** Selecting the symbol vector of (o, i), the objects of a scene and the
   scene of a query are joins with 1.0-valued indicator relations, and each gathered element is
   counted as a ⊗ (ratio_mul 1.225 instead of ≈1.0 at small). A hardware gather (or a
   join mode that copies rather than multiplies) removes this overhead entirely.
2. **No grouped SELECT.** The per-query argmax is Q filters over the whole SIM relation
   (Q·16 tuples each) plus Q `select_k(1)` calls: Q²·16 + Q·16 candidates
   (16,896 small, 264,192 large), quadratic in Q although each query only
   needs its own 16 entries. A `select_k(R, k, by=(q,))` — a row-wise argmax — is what the
   hardware would do in one pass over SIM.
3. **The cleanup is a plain dense GEMM.** SIM = U (Q×d) · CBᵀ (d×16), or rather four GEMMs
   (Q_i × d) × (d × 16), one per attribute, since a query compares only against its attribute's
   codebook; the trace records it as one combine with 16 candidates per U tuple. Nothing in the
   relational form adds work here, and nothing in it exploits the structure either (bipolar keys,
   the fact that the same scene is unbound by several queries).
4. **Zero suppression is a hazard, not a feature.** With bipolar symbols, bundles of four ±1
   terms are exactly zero in a large fraction of coordinates and the relation layer would drop
   them, silently reducing the counted MACs below Q·16·d; Gaussian symbols were chosen to keep the
   count honest. A dense hardware GEMM would not skip those coordinates.

## 8. Shape seen by the hardware
Dense operands: CB is 64 × d (small: 131,072 values, large: 524,288), shared by every query
— the weight-stationary operand of the GEMM; U is Q × d, fully dense; SIM is Q × 16. The Q
dimension shares the codebook (multi-query), and the S scene vectors are shared by the queries
that address the same scene (Q/S ≈ 4 small, ≈ 12 large). Encoding is a chain of elementwise
(bind) and row-sum (bundle) operations on d-length vectors — a vector unit, not a GEMM. Output
does not grow; the GEMM combine dominates (61.8% / 77.4% of ⊗). No rounds.
