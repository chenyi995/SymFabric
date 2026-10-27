from pathlib import Path
exec(compile(Path(__file__).with_name('bench_multi.py').read_text().split('allrows=[]')[0],__file__,'exec'))
import cProfile,pstats
model.set_engine(ExactEngine(model),cache=True,semiring=InferenceScalarSemiring)
vs=[Term('p'+str(i)) for i in range(4)];q=Query(Term('multi_addition',list2term(vs[:2]),list2term(vs[2:]),Var('S')),{v:Term('tensor',Term('images',Constant(i))) for i,v in enumerate(vs)})
with torch.no_grad():
 model.solve([q]);pr=cProfile.Profile();pr.enable();model.solve([q]);pr.disable()
pr.dump_stats(str(b/'scalar_warm.pstats'))
with (b/'scalar_warm_profile.txt').open('w') as f:pstats.Stats(pr,stream=f).sort_stats('cumulative').print_stats(50)
