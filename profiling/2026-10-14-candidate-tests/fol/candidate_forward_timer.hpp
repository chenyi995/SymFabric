#pragma once
#include <chrono>
#include <cstdio>
#include <cstdint>
namespace nesy_forward_diag {
struct Entry {uint64_t calls=0,nested=0,ns=0,maxns=0;};
inline Entry entries[2];
inline uint64_t outer_ns=0;
inline thread_local unsigned depth=0;
struct Scope {
 unsigned id; bool outer;
 std::chrono::steady_clock::time_point start;
 explicit Scope(unsigned i):id(i),outer(depth++==0),start(std::chrono::steady_clock::now()) {entries[id].calls++;if(!outer)entries[id].nested++;}
 ~Scope(){
  auto dt=(uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  auto &e=entries[id];e.ns+=dt;if(dt>e.maxns)e.maxns=dt;
  if(outer)outer_ns+=dt;--depth;
 }
};
inline void emit(){
 const char* names[]={"CodeTreeForwardSubsumptionAndResolution","ForwardSubsumptionAndResolution"};
 for(unsigned i=0;i<2;i++){
  auto &e=entries[i];
  std::fprintf(stderr,"NESY_FORWARD {\"engine\":\"%s\",\"calls\":%llu,\"nested_calls\":%llu,\"inclusive_ns\":%llu,\"max_ns\":%llu}\n",names[i],(unsigned long long)e.calls,(unsigned long long)e.nested,(unsigned long long)e.ns,(unsigned long long)e.maxns);
 }
 std::fprintf(stderr,"NESY_FORWARD_TOTAL {\"outer_ns\":%llu}\n",(unsigned long long)outer_ns);
}
}
