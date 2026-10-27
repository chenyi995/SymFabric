// Thin library caller. No RNNLogic model and no accelerator timing model.
// Example loop follows upstream src/ramulator/python/bindings.cpp.
#include <ramulator/base/config.h>
#include <ramulator/base/factory.h>
#include <ramulator/base/request.h>
#include <ramulator/frontend/i_frontend.h>
#include <ramulator/memory_system/i_memory_system.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Ramulator;
int main(int argc, char** argv) {
  if (argc != 3 && argc != 5) {
    std::cerr << "Usage: ramulator_driver CONFIG example | CONFIG external WINDOW N\n";
    return 2;
  }
  try {
    auto cfg=Config::parse_config_file(argv[1]);
    std::unique_ptr<IFrontEnd> fe(Factory::create_frontend(cfg));
    std::unique_ptr<IMemorySystem> mem(Factory::create_memory_system(cfg));
    fe->connect_memory_system(mem.get()); mem->connect_frontend(fe.get());
    const std::string mode=argv[2];
    uint64_t cycles=0;
    if (mode=="example") {
      const int fe_tick=fe->get_clock_ratio(), mem_tick=mem->get_clock_ratio();
      if (fe_tick<=0 || mem_tick<=0) throw std::runtime_error("bad clock ratio");
      int fe_count=mem_tick-1, mem_count=fe_tick-1;
      uint64_t global_ticks=0;
      while (++global_ticks<100000000) {
        if (++fe_count>=mem_tick) {fe_count=0;fe->tick();}
        if (fe->is_finished()) break;
        if (++mem_count>=fe_tick) {mem_count=0;mem->tick();++cycles;}
      }
      if (!fe->is_finished()) throw std::runtime_error("example timeout");
      std::cout << "driver_example_completed: true\n";
    } else if(mode=="external" && argc==5) {
      const size_t window=std::stoull(argv[3]), n=std::stoull(argv[4]);
      if (!window || !n) throw std::runtime_error("window and request count must be positive");
      size_t accepted=0,completed=0,outstanding=0,peak=0,rejected=0;
      std::vector<bool> seen(n,false);
      const int bytes=mem->get_tx_bytes();
      if(bytes<=0) throw std::runtime_error("bad transaction size");
      while(completed<n && cycles<10000000) {
        while(accepted<n && outstanding<window) {
          const size_t id=accepted;
          // Sparse cache-line stream; synthetic interface check, not a measured workload.
          const uint64_t addr=(static_cast<uint64_t>(id)*4099 % 65536)*bytes;
          bool ok=fe->receive_external_requests(Request::Type::Read,addr,0,
             [&,id](Request& req){
               if(seen[id] || outstanding==0 || req.depart<req.arrive)
                 throw std::runtime_error("bad completion");
               seen[id]=true;++completed;--outstanding;
             },bytes);
          if(!ok) {++rejected;break;}
          ++accepted;++outstanding;peak=std::max(peak,outstanding);
        }
        mem->tick();++cycles;
      }
      if(accepted!=n || completed!=n || outstanding!=0 ||
         !std::all_of(seen.begin(),seen.end(),[](bool x){return x;}))
        throw std::runtime_error("external smoke incomplete");
      std::cout << "driver_external_completed: true\n"
                << "driver_request_count: " << n << "\n"
                << "driver_window: " << window << "\n"
                << "driver_max_outstanding: " << peak << "\n"
                << "driver_rejected_retries: " << rejected << "\n"
                << "driver_transaction_bytes: " << bytes << "\n";
    } else {throw std::runtime_error("unknown mode");}
    fe->finalize();mem->finalize();fe->update_stats_recursive();mem->update_stats_recursive();
    std::cout << "driver_memory_ticks: " << cycles << "\n"
              << "driver_dram_tCK_ns: " << mem->get_tCK() << "\n";
    fe->print_stats(std::cout);mem->print_stats(std::cout);
  } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
