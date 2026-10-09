"""Vector-symbolic architecture attribute query with codebook cleanup (NVSA / Raven style).

Four attributes with 16 symbols each; every symbol is a d-dimensional hypervector. Attribute keys
and object-slot keys are bipolar (+-1), so a key is its own inverse under elementwise multiply.
    object  E(o)   = SUM_i  key(i)  * sym(i, v(o, i))           bind = elementwise product
    scene   SC(s)  = SUM_j  slot(j) * E(o(s, j))                 bundle = elementwise sum
    query   (s, j, i): U = SC(s) * slot(j) * key(i)              unbind with the (slot, attribute) key
    cleanup sim(k) = <U, sym(i, k)> for the 16 symbols of attribute i; answer = argmax_k

Algebra program in REAL (+, x), all encoding steps included:
    G     = combine(OBJ(o,i,k)[1], CB(i,k,d))                   # gather the symbol vectors (o,i,k,d)
    E     = combine(G, KEY(i,d), keep=(o,d))                    # bind with the attribute key, bundle over i
    G2    = combine(SCN(s,j,o)[1], E(o,d))                      # gather objects per scene (s,j,o,d)
    SC    = combine(G2, SLOT(j,d), keep=(s,d))                  # bind with the slot key, bundle over j
    QK    = combine(SLOT(j,d), KEY(i,d))                        # composite unbind keys (j,d,i)
    G3    = combine(QRY(q,s,j,i)[1], SC(s,d))                   # gather the scene per query (q,s,j,i,d)
    U     = combine(G3, QK, keep=(q,i,d))                       # unbind: one product per (q,d)
    SIM   = combine(U(q,i,d), CB(i,k,d), keep=(q,k))            # the cleanup GEMM: Q x 16 x d MACs
    per q: answer(q) = select_k(filter(SIM, key q), 1)          # per-query argmax

Golden: numpy (elementwise products, sums, and CB[i] @ u per query, argmax with first-index ties).
Both sides count multiply-accumulates of the similarity search (Q x 16 x d) plus elementwise binds
(object encoding O x 4 x d, scene encoding S x J x d, composite keys J x 4 x d, unbind Q x d).
"""
from __future__ import annotations

import numpy as np

from algebra import REAL, Relation, combine, select_k, filter as afilter, current_trace

NAME = "vsa_cleanup"
CATEGORY = "vsa"
SEMIRINGS = ["real"]
STRUCTURE = "dag"
HOTSPOT_EVENT = ("multiply-accumulates of the codebook similarity search (Q x 16 x d) plus "
                 "elementwise binds of object, scene and query encoding")
N_ATTR, N_SYM = 4, 16
SCALES = {"small": {"d": 2048, "Q": 32, "S": 8, "J": 3},
          "large": {"d": 8192, "Q": 128, "S": 11, "J": 3}}


def make_instance(scale: str, seed: int):
    p = SCALES[scale]
    rng = np.random.default_rng(seed)
    d, Q, S, J = p["d"], p["Q"], p["S"], p["J"]
    # claude: decision start — symbol hypervectors are Gaussian and keys are bipolar: bipolar keys
    # are self-inverse (the unbind is one elementwise product), and Gaussian symbols keep every
    # bundled coordinate non-zero, so the relation layer's zero suppression cannot skip work and
    # the algebra's MAC count is exactly Q x 16 x d. Objects are bound to a slot key inside a
    # scene so that a query (scene, slot, attribute) has one ground-truth symbol.
    CB = rng.standard_normal((N_ATTR, N_SYM, d))
    KEY = rng.choice([-1.0, 1.0], size=(N_ATTR, d))
    SLOT = rng.choice([-1.0, 1.0], size=(J, d))
    O = S * J                                              # object o = s * J + j
    obj_attr = rng.integers(0, N_SYM, size=(O, N_ATTR))
    triples = [(s, j, i) for s in range(S) for j in range(J) for i in range(N_ATTR)]
    assert Q <= len(triples)
    queries = [triples[t] for t in rng.choice(len(triples), size=Q, replace=False)]
    # claude: decision end
    truth = [int(obj_attr[s * J + j, i]) for (s, j, i) in queries]
    return {"d": d, "Q": Q, "S": S, "J": J, "CB": CB, "KEY": KEY, "SLOT": SLOT,
            "obj_attr": obj_attr, "queries": queries, "truth": truth}


def native(inst):
    d, Q, S, J = inst["d"], inst["Q"], inst["S"], inst["J"]
    CB, KEY, SLOT, obj_attr, queries = (inst["CB"], inst["KEY"], inst["SLOT"], inst["obj_attr"],
                                        inst["queries"])
    O = S * J
    E = np.zeros((O, d))
    for o in range(O):
        for i in range(N_ATTR):
            E[o] += KEY[i] * CB[i, obj_attr[o, i]]
    SC = np.zeros((S, d))
    for s in range(S):
        for j in range(J):
            SC[s] += SLOT[j] * E[s * J + j]
    QK = SLOT[:, None, :] * KEY[None, :, :]                # (J, N_ATTR, d)
    sims = np.empty((Q, N_SYM))
    answers = []
    for q, (s, j, i) in enumerate(queries):
        u = SC[s] * QK[j, i]
        sims[q] = CB[i] @ u
        answers.append(int(np.argmax(sims[q])))           # first index on ties
    binds = O * N_ATTR * d + S * J * d + J * N_ATTR * d + Q * d
    macs = Q * N_SYM * d
    native.last_info = {"binds": binds, "macs": macs,
                        "accuracy": sum(a == t for a, t in zip(answers, inst["truth"])) / Q}
    return {"answers": answers, "sims": sims}, binds + macs


def algebra(inst):
    d, Q, S, J = inst["d"], inst["Q"], inst["S"], inst["J"]
    CB, KEY, SLOT, obj_attr, queries = (inst["CB"], inst["KEY"], inst["SLOT"], inst["obj_attr"],
                                        inst["queries"])
    O = S * J
    tr = current_trace()
    CBr = Relation.dense(("i", "k", "d"), REAL, CB, {"i": N_ATTR, "k": N_SYM, "d": d})
    KEYr = Relation.dense(("i", "d"), REAL, KEY, {"i": N_ATTR, "d": d})
    SLOTr = Relation.dense(("j", "d"), REAL, SLOT, {"j": J, "d": d})
    # indicator relations (value 1.0): which symbol each object has, which object sits in each
    # scene slot, and which (scene, slot, attribute) each query asks for
    OBJr = Relation(("o", "i", "k"), REAL,
                    {(o, i, int(obj_attr[o, i])): 1.0 for o in range(O) for i in range(N_ATTR)},
                    {"o": O, "i": N_ATTR, "k": N_SYM})
    SCNr = Relation(("s", "j", "o"), REAL, {(s, j, s * J + j): 1.0 for s in range(S) for j in range(J)},
                    {"s": S, "j": J, "o": O})
    QRYr = Relation(("q", "s", "j", "i"), REAL, {(q, s, j, i): 1.0 for q, (s, j, i) in enumerate(queries)},
                    {"q": Q, "s": S, "j": J, "i": N_ATTR})
    # claude: decision start — gathers (symbol vector of an object, objects of a scene, scene of a
    # query) are joins with 1.0-valued indicator relations because the algebra has no INDEX/gather
    # operator; the trace counts each gathered element as one mul.
    with tr.tag("encode_objects"):
        G = combine(OBJr, CBr)                              # (o, i, k, d)
        E = combine(G, KEYr, keep=("o", "d"))               # bind with key(i), bundle over i
    with tr.tag("encode_scenes"):
        G2 = combine(SCNr, E)                               # (s, j, o, d)
        SC = combine(G2, SLOTr, keep=("s", "d"))            # bind with slot(j), bundle over j
    with tr.tag("query_keys"):
        QK = combine(SLOTr, KEYr)                           # (j, d, i)
    with tr.tag("unbind"):
        G3 = combine(QRYr, SC)                              # (q, s, j, i, d)
        U = combine(G3, QK, keep=("q", "i", "d"))           # unbind
    # claude: decision end
    with tr.tag("cleanup_gemm"):
        SIM = combine(U, CBr, keep=("q", "k"))              # (q, k): Q x 16 x d MACs
    # claude: decision start — the algebra has no grouped select, so the per-query argmax is one
    # filter over SIM (Q x 16 tuples) followed by select_k(1) for each of the Q queries.
    answers = []
    with tr.tag("select"):
        for q in range(Q):
            Fq = afilter(SIM, lambda k, v, q=q: k[0] == q, name="eq_query")
            best = select_k(Fq, 1)
            answers.append(next(iter(best.keys()))[1])
    # claude: decision end
    sims = np.array([[SIM.get((q, k), 0.0) for k in range(N_SYM)] for q in range(Q)])
    phases = {}
    for r in tr.records:
        ph = phases.setdefault(r.tag, {"candidates": 0, "mul_ops": 0, "add_ops": 0})
        ph["candidates"] += r.candidates
        ph["mul_ops"] += r.mul_ops
        ph["add_ops"] += r.add_ops
    algebra.last_info = {"u_nnz": len(U), "sim_nnz": len(SIM), "phases": phases,
                         "accuracy": sum(a == t for a, t in zip(answers, inst["truth"])) / Q}
    return {"answers": answers, "sims": sims}


def compare(a, b) -> bool:
    return (a["answers"] == b["answers"]
            and np.allclose(a["sims"], b["sims"], rtol=1e-9, atol=1e-6))


def extra_info(inst, res):
    li, ni = algebra.last_info, native.last_info
    d, Q = inst["d"], inst["Q"]
    ph = li["phases"]
    return {
        "d": d, "Q": Q, "objects": inst["S"] * inst["J"],
        "native_macs": ni["macs"], "native_binds": ni["binds"],
        "accuracy_native": ni["accuracy"], "accuracy_algebra": li["accuracy"],
        "U_density": li["u_nnz"] / (Q * d),
        "gemm_mul_ops": ph["cleanup_gemm"]["mul_ops"], "gemm_add_ops": ph["cleanup_gemm"]["add_ops"],
        "bind_phase_mul_ops": {k: v["mul_ops"] for k, v in ph.items()
                               if k in ("encode_objects", "encode_scenes", "query_keys", "unbind")},
        # select phase: Q filters over SIM (Q*16 tuples each) + Q select_k(1) calls
        "select_phase_candidates": ph["select"]["candidates"],
        "control_residue": "none (loop over the Q queries only)",
    }
