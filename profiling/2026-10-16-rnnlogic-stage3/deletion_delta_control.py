"""Targeted exact CPU control: relation absent, or once at body position zero.

This is a materializing sparse-terminal wrapper over the existing SPA backend.
It is not a generic repeated-relation delta engine or a neural predictor. All
subtraction precedes float conversion; unsupported rule shapes are rejected.
"""
import time
import numpy as np
import torch
from official_native_adapter import cpu_i64


class FirstEdgeDelta:
    def __init__(self, backend, relation, bodies, heads):
        start = time.perf_counter()
        self.backend = backend
        self.relation = int(relation)
        self.entities = backend.entity_size
        self.bodies = [tuple(map(int, body)) for body in bodies]
        self.heads = np.unique(cpu_i64(heads))
        self.head_to_slot = {int(head): slot for slot, head in enumerate(self.heads)}
        self.affected = {}
        suffixes = []
        suffix_to_slot = {}
        for rule, body in enumerate(self.bodies):
            occurrences = [i for i, r in enumerate(body) if r == self.relation]
            if occurrences and occurrences != [0]:
                raise ValueError('only absent or first-position-once query relation is supported')
            if occurrences:
                suffix = body[1:]
                if suffix not in suffix_to_slot:
                    suffix_to_slot[suffix] = len(suffixes)
                    suffixes.append(suffix)
                self.affected[rule] = suffix_to_slot[suffix]
        # Separate proofs include every helper prefix, not only full terminals.
        self.base_plan = backend.prepare_rules(self.bodies)
        self.suffix_plan = backend.prepare_rules(suffixes)
        self.plan_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw = backend.batch(self.base_plan, self.heads, self.relation, None)
        self.base_count_seconds = time.perf_counter() - start
        start = time.perf_counter()
        self.rows = []
        self.owners = []
        for index, value in raw:
            flat = index.numpy()
            boundaries = np.searchsorted(flat, np.arange(len(self.heads) + 1, dtype=np.int64) * self.entities)
            ids = flat % self.entities
            counts = value.numpy().copy()
            self.owners.extend((ids, counts, boundaries))
            self.rows.append([(ids[lo:hi], counts[lo:hi]) for lo, hi in zip(boundaries[:-1], boundaries[1:])])
        del raw
        self.base_layout_seconds = time.perf_counter() - start
        self.base_storage_bytes = sum(array.nbytes for array in self.owners)
        self.base_nnz = sum(len(self.owners[i]) for i in range(0, len(self.owners), 3))

    def batch(self, heads, tails, removed):
        start = time.perf_counter()
        h, t, removed = cpu_i64(heads), cpu_i64(tails), cpu_i64(removed)
        if h.ndim != 1 or t.shape != h.shape or removed.shape != h.shape:
            raise ValueError('query shape')
        source, destination = self.backend.adjacency[self.relation]
        if removed.size and (removed.min() < 0 or removed.max() >= len(source)):
            raise ValueError('removed edge range')
        if not np.array_equal(source[removed], h) or not np.array_equal(destination[removed], t):
            raise ValueError('deleted edge must be the actual query (head, relation, tail)')
        slots = [self.head_to_slot[int(head)] for head in h]
        unique_tails, tail_slots = np.unique(t, return_inverse=True)
        preparation_seconds = time.perf_counter() - start
        start = time.perf_counter()
        suffixes = self.backend.batch(self.suffix_plan, unique_tails, self.relation, None)
        suffix_seconds = time.perf_counter() - start
        start = time.perf_counter()
        suffix_rows = []
        for index, value in suffixes:
            flat = index.numpy()
            boundaries = np.searchsorted(flat, np.arange(len(unique_tails) + 1, dtype=np.int64) * self.entities)
            ids = flat % self.entities
            counts = value.numpy()
            suffix_rows.append([(ids[lo:hi], counts[lo:hi]) for lo, hi in zip(boundaries[:-1], boundaries[1:])])
        outputs = []
        corrected_to_zero = 0
        for rule, base_rows in enumerate(self.rows):
            flat_chunks, value_chunks = [], []
            for query, base_slot in enumerate(slots):
                ids, counts = base_rows[base_slot]
                if rule in self.affected:
                    delta_ids, delta_counts = suffix_rows[self.affected[rule]][tail_slots[query]]
                    corrected = counts.copy()
                    if len(delta_ids):
                        positions = np.searchsorted(ids, delta_ids)
                        if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], delta_ids):
                            raise AssertionError('suffix support is not a subset of unmasked base')
                        if np.any(corrected[positions] < delta_counts):
                            raise AssertionError('deletion count exceeds unmasked base')
                        corrected[positions] -= delta_counts
                    keep = corrected > 0
                    corrected_to_zero += int(len(corrected) - np.count_nonzero(keep))
                    flat_chunks.append(ids[keep] + query * self.entities)
                    value_chunks.append(corrected[keep])
                else:
                    flat_chunks.append(ids + query * self.entities)
                    value_chunks.append(counts)
            # Concatenate even unchanged columns: output contract and copies
            # match native batch terminals, rather than passing hidden views.
            index = np.concatenate(flat_chunks) if flat_chunks else np.empty(0, dtype=np.int64)
            value = np.concatenate(value_chunks) if value_chunks else np.empty(0, dtype=np.int64)
            outputs.append((torch.from_numpy(index), torch.from_numpy(value)))
        materialization_seconds = time.perf_counter() - start
        self.last = {'query_validation_and_lookup_seconds': preparation_seconds,
                     'suffix_count_seconds': suffix_seconds,
                     'merge_and_materialization_seconds': materialization_seconds,
                     'unique_tails': len(unique_tails),
                     'suffix_nonzeros': sum(len(index) for index, _ in suffixes),
                     'corrected_entries_removed': corrected_to_zero}
        return outputs
