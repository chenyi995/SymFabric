// Local standard dense SPA/touched-list control for exact int64 rule counts.
// Compile beside unchanged upstream miner/rnnlogic.cpp. Independent rules and
// original vector[entity][relation] neighbor lists; no prefix/count memoization.
#include "rnnlogic.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <omp.h>

struct SPAWorkspace {
    std::vector<int64_t> count[2];
    std::vector<int> touched[2];
    explicit SPAWorkspace(int entities) {
        for(int side=0;side<2;++side) {
            count[side].assign(entities,0);
            touched[side].reserve(entities);
        }
    }
    void clear(int side) {
        for(int entity:touched[side]) count[side][entity]=0;
        touched[side].clear();
    }
};

class SPAAdapter : public KnowledgeGraph {
public:
    SPAAdapter(int n,int nr,const int64_t* offsets,const int64_t* src,const int64_t* dst) {
        auto rows=new std::vector<int>*[n]();
        int allocated=0;
        try {
            for(;allocated<n;++allocated) rows[allocated]=new std::vector<int>[nr];
            for(int relation=0;relation<nr;++relation)
                for(int64_t edge=offsets[relation];edge<offsets[relation+1];++edge)
                    rows[src[edge]][relation].push_back(static_cast<int>(dst[edge]));
        } catch(...) {
            for(int entity=0;entity<allocated;++entity) delete[] rows[entity];
            delete[] rows;throw;
        }
        e2r2n=rows;entity_size=n;relation_size=nr;
    }
    using Terminal=std::vector<std::pair<int,int64_t>>;
    void destination(int head,const Rule& rule,Triplet removed,SPAWorkspace& workspace,Terminal& output) {
        // Previous rule leaves both buffers/touched lists empty. Counts are
        // positive, so zero means unseen without another per-entity marker.
        int current=0;
        workspace.count[current][head]=1;
        workspace.touched[current].push_back(head);
        for(int relation:rule.r_body) {
            int next=1-current;
            auto& input=workspace.count[current];
            auto& accum=workspace.count[next];
            auto& active=workspace.touched[next];
            for(int entity:workspace.touched[current]) {
                const int64_t paths=input[entity];
                for(int destination:e2r2n[entity][relation]) {
                    if(entity==removed.h && relation==removed.r && destination==removed.t) continue;
                    if(accum[destination]==0) active.push_back(destination);
                    // Same all-query intermediate-count proof as native64.
                    accum[destination]+=paths;
                }
            }
            workspace.clear(current);
            current=next;
            if(workspace.touched[current].empty()) break;
        }
        auto& active=workspace.touched[current];
        std::sort(active.begin(),active.end());
        output.reserve(active.size());
        for(int entity:active) output.emplace_back(entity,workspace.count[current][entity]);
        workspace.clear(current);
    }
};

struct SPAResult {std::vector<int64_t> offsets,flat,values;};
extern "C" {
void* official_graph_create(int n,int nr,const int64_t* offsets,const int64_t* src,const int64_t* dst) {
    try{return new SPAAdapter(n,nr,offsets,src,dst);}catch(...){return nullptr;}
}
void official_graph_destroy(void* graph){delete static_cast<SPAAdapter*>(graph);}
void* official_ground_batch(void* pointer,const int64_t* heads,int64_t batch,
 const int64_t* offsets,const int64_t* body,int64_t nrules,
 const int64_t* removed_h,const int64_t* removed_r,const int64_t* removed_t,int bits,int threads) {
    if(bits!=64 || threads<1)return nullptr;
    try {
        auto* graph=static_cast<SPAAdapter*>(pointer);
        std::vector<Rule> rules(nrules);
        for(int64_t rule=0;rule<nrules;++rule)
            for(int64_t index=offsets[rule];index<offsets[rule+1];++index)
                rules[rule].r_body.push_back(static_cast<int>(body[index]));
        std::vector<SPAAdapter::Terminal> outputs(nrules*batch);
        std::vector<std::unique_ptr<SPAWorkspace>> workspaces(threads);
        std::atomic<bool> failed(false);
        #pragma omp parallel num_threads(threads)
        {
            int tid=omp_get_thread_num();
            try{workspaces[tid]=std::make_unique<SPAWorkspace>(graph->get_entity_size());}
            catch(...){failed.store(true);}
            #pragma omp barrier
            #pragma omp for schedule(static)
            for(int64_t query=0;query<batch;++query) {
                if(failed.load())continue;
                try {
                    Triplet removed{static_cast<int>(removed_h[query]),static_cast<int>(removed_t[query]),static_cast<int>(removed_r[query])};
                    for(int64_t rule=0;rule<nrules;++rule)
                        graph->destination(static_cast<int>(heads[query]),rules[rule],removed,*workspaces[tid],outputs[rule*batch+query]);
                }catch(...){failed.store(true);}
            }
        }
        if(failed.load())return nullptr;
        auto result=std::make_unique<SPAResult>();
        size_t total=0;for(const auto& output:outputs)total+=output.size();
        result->flat.reserve(total);result->values.reserve(total);result->offsets.push_back(0);
        for(int64_t rule=0;rule<nrules;++rule) {
            for(int64_t query=0;query<batch;++query)
                for(auto [entity,count]:outputs[rule*batch+query]) {
                    result->flat.push_back(query*graph->get_entity_size()+entity);
                    result->values.push_back(count);
                }
            result->offsets.push_back(result->flat.size());
        }
        return result.release();
    }catch(...){return nullptr;}
}
int64_t official_result_nnz(void* result){return static_cast<SPAResult*>(result)->flat.size();}
const int64_t* official_result_offsets(void* result){return static_cast<SPAResult*>(result)->offsets.data();}
const int64_t* official_result_flat(void* result){return static_cast<SPAResult*>(result)->flat.data();}
const int64_t* official_result_values(void* result){return static_cast<SPAResult*>(result)->values.data();}
void official_result_destroy(void* result){delete static_cast<SPAResult*>(result);}
}
