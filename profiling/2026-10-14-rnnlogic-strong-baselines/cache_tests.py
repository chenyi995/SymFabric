"""Small independent correctness checks; no dataset, GPU, or scatter dependency."""
import unittest

import torch

from prefix_cache import PrefixGrounder


class TinyGraph:
    entity_size = 5
    edges = {
        0: [(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)],
        1: [(1, 4), (2, 4), (4, 0)],
        2: [],
    }

    def propagate(self, x, relation, removed=None):
        result = torch.zeros_like(x)
        edges = self.edges[relation]
        if not edges:
            return result
        src = torch.tensor([edge[0] for edge in edges], device=x.device)
        dst = torch.tensor([edge[1] for edge in edges], device=x.device)
        message = x[src].clone()
        if removed is not None:
            message[removed, torch.arange(x.shape[1], device=x.device)] = 0
        result.index_add_(0, dst, message)
        return result


def reference(graph, heads, head_relation, body, removed):
    # Python integers and explicit per-query edge traversal are independent of
    # tensor gather/scatter and of the prefix controller.
    output = []
    for query, head in enumerate(heads.tolist()):
        counts = [int(entity == head) for entity in range(graph.entity_size)]
        for relation in body:
            next_counts = [0] * graph.entity_size
            for edge_index, (src, dst) in enumerate(graph.edges[relation]):
                if relation == head_relation and removed is not None and edge_index == removed[query].item():
                    continue
                next_counts[dst] += counts[src]
            counts = next_counts
        output.append(counts)
    return torch.tensor(output, dtype=torch.int64, device=heads.device)


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.graph = TinyGraph()
        self.h = torch.tensor([0, 3, 1])
        self.removed = torch.tensor([0, 4, 2])
        self.bodies = [(), (0,), (0, 0), (0, 1), (0, 0), (0, 1, 0), (1, 0), (0, 0, 0), (2,), ()]
        self.state_bytes = self.h.numel() * self.graph.entity_size * 8
        self.unique = {tuple(b[:i]) for b in self.bodies for i in range(1, len(b) + 1)}

    def run_forward(self, grounder, removed):
        grounder.begin(self.h, 0, removed)
        for body in self.bodies:
            result = grounder.grounding(self.h, 0, body, removed)
            self.assertEqual(result.dtype, torch.int64)
            self.assertFalse(result.requires_grad)
            self.assertTrue(torch.equal(result, reference(self.graph, self.h, 0, body, removed)))
        return grounder.finish()

    def test_all_policies_with_and_without_per_query_removal(self):
        for policy, budget in [("all", None), ("last_use", None), ("lru", 0), ("lru", self.state_bytes - 1), ("lru", self.state_bytes), ("lru", self.state_bytes * 2), ("lru", self.state_bytes * 20), ("last_use_lru", 0), ("last_use_lru", self.state_bytes - 1), ("last_use_lru", self.state_bytes), ("last_use_lru", 2 * self.state_bytes), ("last_use_lru", 20 * self.state_bytes)]:
            for removed in [None, self.removed]:
                with self.subTest(policy=policy, budget=budget, removed=removed is not None):
                    grounder = PrefixGrounder(self.graph, self.bodies, policy, budget)
                    stats = self.run_forward(grounder, removed)
                    self.assertTrue(stats["complete"])
                    self.assertEqual(stats["rule_calls"], len(self.bodies))
                    self.assertGreaterEqual(stats["max_live_bytes"], stats["max_cache_bytes"])
                    self.assertEqual(grounder.stats["current_cache_bytes"], 0)
                    if policy in {"lru", "last_use_lru"}:
                        self.assertLessEqual(stats["max_cache_bytes"], budget)
                        self.assertLessEqual(stats["max_live_bytes"], budget + 2 * self.state_bytes)
                    else:
                        self.assertEqual(stats["propagations"], len(self.unique))
                        self.assertEqual(stats["recomputes"], 0)
                    if budget is not None and budget < self.state_bytes:
                        self.assertEqual(stats["propagations"], sum(map(len, self.bodies)))
                        self.assertEqual(stats["max_cache_bytes"], 0)
                        self.assertEqual(stats["max_live_bytes"], 2 * self.state_bytes)

    def test_multiplicity_and_query_specific_removal(self):
        body = (0, 0)
        heads = torch.tensor([0, 0])
        removed = torch.tensor([0, 1])
        cache = PrefixGrounder(self.graph, [body])
        cache.begin(heads, 0, None)
        result = cache.grounding(heads, 0, body, None)
        self.assertEqual(result[:, 3].tolist(), [2, 2])
        cache.finish()
        cache.begin(heads, 0, removed)
        result = cache.grounding(heads, 0, body, removed)
        self.assertEqual(result[:, 3].tolist(), [1, 1])
        cache.finish()

    def test_last_use_reduces_storage_without_recomputation(self):
        all_stats = self.run_forward(PrefixGrounder(self.graph, self.bodies, "all"), self.removed)
        last_stats = self.run_forward(PrefixGrounder(self.graph, self.bodies, "last_use"), self.removed)
        self.assertLess(last_stats["max_cache_bytes"], all_stats["max_cache_bytes"])
        self.assertEqual(last_stats["final_cache_bytes"], 0)
        self.assertEqual(all_stats["max_cache_bytes"], (len(self.unique) + 1) * self.state_bytes)
        self.assertEqual(last_stats["propagations"], all_stats["propagations"])

    def test_last_use_lru_does_not_evict_reusable_prefix_for_dead_terminal(self):
        # With room for one state, plain LRU stores the first terminal and evicts
        # its reusable parent. The combined policy retains that parent instead.
        bodies = [(0, 0), (0, 1)]
        results = {}
        for policy in ("lru", "last_use_lru"):
            cache = PrefixGrounder(self.graph, bodies, policy, self.state_bytes)
            cache.begin(self.h, 0, self.removed)
            for body in bodies:
                actual = cache.grounding(self.h, 0, body, self.removed)
                self.assertTrue(torch.equal(actual, reference(self.graph, self.h, 0, body, self.removed)))
                self.assertLessEqual(cache.stats["current_cache_bytes"], self.state_bytes)
            results[policy] = cache.finish()
        self.assertEqual(results["lru"]["propagations"], 4)
        self.assertEqual(results["last_use_lru"]["propagations"], 3)
        self.assertEqual(results["last_use_lru"]["final_cache_bytes"], 0)

    def test_last_use_lru_with_sufficient_budget_matches_last_use(self):
        unbounded = self.run_forward(PrefixGrounder(self.graph, self.bodies, "last_use"), self.removed)
        combined = self.run_forward(PrefixGrounder(self.graph, self.bodies, "last_use_lru", 20 * self.state_bytes), self.removed)
        for field in ("propagations", "recomputes", "max_cache_bytes", "max_live_bytes", "final_cache_bytes"):
            self.assertEqual(combined[field], unbounded[field])
        self.assertEqual(combined["evictions"], 0)

    def test_begin_resets_queries_and_statistics(self):
        cache = PrefixGrounder(self.graph, self.bodies, "last_use")
        first = self.run_forward(cache, self.removed)
        self.h = torch.tensor([2, 0, 4])
        second = self.run_forward(cache, None)
        self.assertEqual(first["propagations"], second["propagations"])
        self.assertEqual(first["rule_calls"], second["rule_calls"])

    def test_stats_can_be_disabled_without_changing_results_or_capacity(self):
        for policy, budget in [("all", None), ("last_use", None), ("lru", 0), ("lru", self.state_bytes), ("lru", 2 * self.state_bytes), ("last_use_lru", 0), ("last_use_lru", self.state_bytes), ("last_use_lru", 2 * self.state_bytes)]:
            with self.subTest(policy=policy, budget=budget):
                cache = PrefixGrounder(self.graph, self.bodies, policy, budget, collect_stats=False)
                cache.begin(self.h, 0, self.removed)
                for body in self.bodies:
                    output = cache.grounding(self.h, 0, body, self.removed)
                    self.assertTrue(torch.equal(output, reference(self.graph, self.h, 0, body, self.removed)))
                    if policy in {"lru", "last_use_lru"}:
                        self.assertLessEqual(cache.stats["current_cache_bytes"], budget)
                stats = cache.finish()
                self.assertFalse(stats["collect_stats"])
                self.assertEqual(stats["max_live_bytes"], 0)
                self.assertEqual(stats["propagations"], 0)
                cache.collect_stats = True
                enabled = self.run_forward(cache, self.removed)
                self.assertTrue(enabled["collect_stats"])
                self.assertGreater(enabled["max_live_bytes"], 0)
                self.assertGreater(enabled["propagations"], 0)

    def test_wrong_order_or_new_query_object_is_rejected(self):
        cache = PrefixGrounder(self.graph, self.bodies)
        cache.begin(self.h, 0, self.removed)
        with self.assertRaises(ValueError):
            cache.grounding(self.h, 0, (0,), self.removed)
        with self.assertRaises(ValueError):
            cache.grounding(self.h.clone(), 0, (), self.removed)
        with self.assertRaises(ValueError):
            cache.grounding(self.h, 1, (), self.removed)
        with self.assertRaises(ValueError):
            cache.grounding(self.h, 0, (), self.removed.clone())
        self.h[0] = 2
        with self.assertRaises(ValueError):
            cache.grounding(self.h, 0, (), self.removed)
        self.assertFalse(cache.finish(require_complete=False)["complete"])

    def test_protocol_errors_and_empty_rule_set(self):
        cache = PrefixGrounder(self.graph, [])
        with self.assertRaises(RuntimeError):
            cache.grounding(self.h, 0, ())
        cache.begin(self.h, 0)
        with self.assertRaises(RuntimeError):
            cache.begin(self.h, 0)
        self.assertTrue(cache.end()["complete"])
        cache = PrefixGrounder(self.graph, [(0,)])
        cache.begin(self.h, 0)
        with self.assertRaises(RuntimeError):
            cache.finish()
        cache.begin(self.h, 0)
        cache.grounding(self.h, 0, (0,))
        cache.finish()


if __name__ == "__main__":
    unittest.main(verbosity=2)
