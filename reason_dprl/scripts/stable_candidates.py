"""Opt-in correction for upstream pack_results' colliding CUDA scatter.

Preserve fast-mode fact-then-rule order and retain the first K valid entries.
The downloaded upstream source is never modified. This adapter must be enabled
identically on both compared backends and disclosed in benchmark provenance.
"""
import torch


def pack_results(fact_states, fact_mask, rule_states, rule_mask, K_max, M_max, padding_idx):
    B=fact_states.shape[0]
    def atoms(x):
        if x.shape[2]<M_max:
            tail=x.new_full((B,x.shape[1],M_max-x.shape[2],3),padding_idx)
            return torch.cat((x,tail),dim=2)
        return x[:,:,:M_max]
    states=torch.cat((atoms(fact_states),atoms(rule_states)),dim=1)
    valid=torch.cat((fact_mask,rule_mask),dim=1).bool()
    K=states.shape[1]
    count=valid.sum(1).clamp(max=K_max)
    if K==0:
        return states.new_full((B,K_max,M_max,3),padding_idx),count
    ranks=valid.long().cumsum(1)-1
    targets=torch.where(valid & (ranks<K_max),ranks,K_max)
    indices=torch.arange(K,device=states.device).expand(B,K)
    # Only the extra, discarded sink slot has duplicate destinations. amin is
    # order-independent there; every retained valid destination has one writer.
    packed=states.new_full((B,K_max+1),K)
    packed.scatter_reduce_(1,targets,indices,reduce="amin",include_self=True)
    selected=packed[:,:K_max]
    gathered=states.gather(1,selected.clamp(max=K-1)[:,:,None,None].expand(B,K_max,M_max,3))
    return torch.where((selected<K)[:,:,None,None],gathered,padding_idx),count


def install():
    import kge_experiments.unification as unification
    unification.pack_results=pack_results
    return {"name":"stable_first_k_candidates", "scope":"unification.pack_results only",
        "order":"fact then rule, same as upstream fast mode",
        "overflow":"first K valid entries; invalid/overflow writes go to a discarded sink",
        "reason":"upstream scatter has concurrent writers at K_max-1 and nondeterministic terminal outputs"}
