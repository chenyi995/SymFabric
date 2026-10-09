/* Execution-contract probe of the official DPLA* Prolog engine.
   This uses a deterministic test heuristic, not a trained neural model.
   Pass the pinned engine_heap.pl as the sole positional argument. */
:- use_module(library(http/json)).
:- use_module(library(terms)).
:- use_module(library(pairs)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- dynamic observed_heuristic/1.

get_heuristic_extern(Goal, 0.8) :- assertz(observed_heuristic(Goal)).

setup_probe :-
    retractall(fa(_,_,_,_)), retractall(cl(_,_,_)),
    assertz(cl(1,path(X,Y),[edge(X,Z),edge(Z,Y)])),
    assertz(fa(1,0.9,edge(a,b),none)),
    assertz(fa(2,0.8,edge(b,c),none)),
    assertz(fa(3,0.7,edge(a,d),none)),
    assertz(fa(4,0.6,edge(d,e),none)).

goal_list(_-n(Pairs,_,_), Goals) :- pairs_values(Pairs,Goals).

case(Query, Dict) :-
    abolish_all_tables, retractall(observed_heuristic(_)),
    extern([edge], H),
    get_all_node_children(H-n([_-Query],_,[]), RuleChildren),
    RuleChildren=[RuleChild],
    goal_list(RuleChild, RuleGoals),
    RuleGoals=[edge(_,V),edge(V,_)], var(V),
    get_all_node_children(RuleChild, FactChildren),
    maplist(goal_list,FactChildren,BoundGoals),
    maplist(term_string,BoundGoals,BoundStrings),
    length(FactChildren, Count),
    maplist(term_size,BoundGoals,Cells),
    findall(G,observed_heuristic(G),Seen),
    maplist(term_string,Seen,SeenStrings),
    term_string(Query,QueryString),
    term_string(RuleGoals,RuleString),
    Dict=_{query:QueryString, rule_children:1, fact_children:Count,
           unresolved_goals:RuleString, bound_goals:BoundStrings,
           bound_goal_term_cells:Cells, heuristic_arguments:SeenStrings}.

main :-
    current_prolog_flag(argv,[Engine]), consult(Engine), setup_probe,
    case(path(a,c),A), case(path(b,c),B),
    A.fact_children=2, B.fact_children=1,
    A.bound_goals=["[edge(b,c)]","[edge(d,c)]"],
    B.bound_goals=["[edge(c,c)]"],
    length(A.heuristic_arguments,2), length(B.heuristic_arguments,1),
    json_write_dict(current_output,_{passed:true,cases:[A,B],
      scope:"Official get_node_child/findall and heuristic invocation; synthetic clauses and deterministic heuristic; no end-to-end latency claim"}),
    nl, halt(0).

:- initialization((main -> true ; halt(1)),main).
