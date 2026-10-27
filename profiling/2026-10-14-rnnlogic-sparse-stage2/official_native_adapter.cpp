// Adapter to RNNLogic c890b025 miner/rnnlogic.cpp:412-444.
// The original .cpp is compiled unchanged beside this translation unit.
// native32 calls the original method. native64 retains its std::map algorithm,
// widens counts to int64, uses standard vector storage instead of a VLA, and
// defines empty-body identity to match Python. No shared-prefix computation.
#include "rnnlogic.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <atomic>
#include <new>

class OfficialAdapter : public KnowledgeGraph {
public:
    OfficialAdapter(int n, int nr, const int64_t* offsets,
                    const int64_t* src, const int64_t* dst) {
        auto rows = new std::vector<int>*[n]();
        int allocated=0;
        try {
            for (;allocated<n;++allocated) rows[allocated]=new std::vector<int>[nr];
            for(int r=0;r<nr;++r)
                for(int64_t i=offsets[r];i<offsets[r+1];++i)
                    rows[src[i]][r].push_back(static_cast<int>(dst[i]));
        } catch(...) {
            for(int i=0;i<allocated;++i) delete[] rows[i];
            delete[] rows; throw;
        }
        e2r2n=rows;entity_size=n;relation_size=nr;
    }

    void rule_destination64(int e, const Rule& rule,
                            std::map<int,int64_t>* dest2count, Triplet removed) {
        std::vector<std::map<int,int64_t>> m(rule.r_body.size()+1);
        std::map<int,int64_t>* pt;
        dest2count->clear();
        m[0][e]=1;
        if(rule.r_body.empty()) {(*dest2count)[e]=1;return;}
        for(size_t k=0;k<rule.r_body.size();++k) {
            if(k==rule.r_body.size()-1) pt=dest2count;
            else pt=&m[k+1];
            for(auto iter=m[k].begin();iter!=m[k].end();++iter) {
                const int current_e=iter->first;
                const int64_t current_n=iter->second;
                const int current_r=rule.r_body[k];
                for(size_t i=0;i<e2r2n[current_e][current_r].size();++i) {
                    const int next_e=e2r2n[current_e][current_r][i];
                    if(current_e==removed.h && current_r==removed.r && next_e==removed.t) continue;
                    if(pt->count(next_e)==0) (*pt)[next_e]=0;
                    // The wrapper proves an all-query path-count bound before calls.
                    (*pt)[next_e]+=current_n;
                }
            }
        }
    }
};

struct NativeResult {
    std::vector<int64_t> offsets,flat,values;
};

extern "C" {
void* official_graph_create(int n,int nr,const int64_t* offsets,const int64_t* src,const int64_t* dst) {
    try {return new OfficialAdapter(n,nr,offsets,src,dst);} catch(...) {return nullptr;}
}
void official_graph_destroy(void* p) {delete static_cast<OfficialAdapter*>(p);}

void* official_ground_batch(void* p,const int64_t* heads,int64_t batch,
    const int64_t* rule_offsets,const int64_t* body,int64_t nrules,
    const int64_t* removed_h,const int64_t* removed_r,const int64_t* removed_t,
    int bits,int threads) {
  try {
    auto* graph=static_cast<OfficialAdapter*>(p);
    std::vector<Rule> rules(nrules);
    for(int64_t k=0;k<nrules;++k)
      for(int64_t j=rule_offsets[k];j<rule_offsets[k+1];++j)
        rules[k].r_body.push_back(static_cast<int>(body[j]));
    using Terminal=std::vector<std::pair<int,int64_t>>;
    std::vector<Terminal> outputs(nrules*batch);
    std::atomic<bool> failed(false);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for(int64_t b=0;b<batch;++b) {
      try {
        Triplet removed{static_cast<int>(removed_h[b]),static_cast<int>(removed_t[b]),static_cast<int>(removed_r[b])};
        for(int64_t k=0;k<nrules;++k) {
          auto& out=outputs[k*batch+b];
          if(bits==32) {
            std::map<int,int> counts;
            graph->rule_destination(static_cast<int>(heads[b]),rules[k],&counts,removed);
            out.reserve(counts.size());
            for(auto [entity,count]:counts) out.emplace_back(entity,count);
          } else {
            std::map<int,int64_t> counts;
            graph->rule_destination64(static_cast<int>(heads[b]),rules[k],&counts,removed);
            out.reserve(counts.size());
            for(auto [entity,count]:counts) out.emplace_back(entity,count);
          }
        }
      } catch(...) {failed.store(true);}
    }
    if(failed.load()) return nullptr;
    auto* result=new NativeResult;
    size_t total=0;
    for(const auto& v:outputs) total+=v.size();
    result->flat.reserve(total);result->values.reserve(total);result->offsets.push_back(0);
    for(int64_t k=0;k<nrules;++k) {
      for(int64_t b=0;b<batch;++b)
        for(auto [entity,count]:outputs[k*batch+b]) {
          result->flat.push_back(b*graph->get_entity_size()+entity);
          result->values.push_back(count);
        }
      result->offsets.push_back(result->flat.size());
    }
    return result;
  } catch(...) {return nullptr;}
}
int64_t official_result_nnz(void* p) {return static_cast<NativeResult*>(p)->flat.size();}
const int64_t* official_result_offsets(void* p) {return static_cast<NativeResult*>(p)->offsets.data();}
const int64_t* official_result_flat(void* p) {return static_cast<NativeResult*>(p)->flat.data();}
const int64_t* official_result_values(void* p) {return static_cast<NativeResult*>(p)->values.data();}
void official_result_destroy(void* p) {delete static_cast<NativeResult*>(p);}
}
