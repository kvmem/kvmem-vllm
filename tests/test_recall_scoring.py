"""Independent scalar oracles for the five fixed-policy recall scorers."""
import math

import pytest
import torch

from vllm_kvmem.recall_scoring import (
    METHODS, build_block_features, block_logits, concat_block_features,
)


def scalar_fixed4(vectors):
    """Direct port of scalar seed/assignment rules, independent of tensor code."""
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]

    def cosine(left, right):
        return (sum(a * b for a, b in zip(left, right))
                / math.sqrt(max(sum(a * a for a in left), 1e-20))
                / math.sqrt(max(sum(a * a for a in right), 1e-20)))

    seeds = [max(range(len(vectors)), key=lambda i: (cosine(vectors[i], mean), -i))]
    best = [cosine(v, vectors[seeds[0]]) for v in vectors]
    assignment = [0] * len(vectors)
    for slot in range(1, 4):
        available = [i for i in range(len(vectors)) if i not in seeds]
        seed = min(available, key=lambda i: (best[i], i)) if available else seeds[0]
        seeds.append(seed)
        for i, vector in enumerate(vectors):
            value = cosine(vector, vectors[seed])
            if value > best[i]:
                best[i], assignment[i] = value, slot
    result = []
    for slot in range(4):
        members = [v for v, assigned in zip(vectors, assignment) if assigned == slot]
        result.append([sum(v[d] for v in members) / len(members) for d in range(dim)]
                      if members else vectors[seeds[0]])
    return result


def scalar_fps4(vectors):
    chosen = []
    for _ in range(min(4, len(vectors))):
        def rank(i):
            value = (min(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j]))
                         for j in chosen) if chosen else sum(v * v for v in vectors[i]))
            return value, -i
        chosen.append(max((i for i in range(len(vectors)) if i not in chosen), key=rank))
    return [vectors[i] for i in chosen]


def scalar_logits(keys, counts, queries, method):
    k, q = keys.double().tolist(), queries.double().tolist()
    layers, blocks, _, kvheads, dim = keys.shape
    rows, heads = queries.shape[1:3]
    result = torch.full((layers, rows, heads, blocks), -torch.inf)
    for layer in range(layers):
        for row in range(rows):
            for head in range(heads):
                query = q[layer][row][head]
                kh = head // (heads // kvheads)
                for block, count in enumerate(counts):
                    if not count:
                        continue
                    vectors = [k[layer][block][token][kh] for token in range(count)]
                    if method == 'mean_k':
                        vectors = [[sum(v[d] for v in vectors) / count for d in range(dim)]]
                    elif method == 'fixed4':
                        vectors = scalar_fixed4(vectors)
                    elif method == 'fps4':
                        vectors = scalar_fps4(vectors)
                    elif method == 'min_max':
                        result[layer, row, head, block] = sum(
                            query[d] * (max(v[d] for v in vectors) if query[d] >= 0 else
                                        min(v[d] for v in vectors)) for d in range(dim)) / math.sqrt(dim)
                        continue
                    scores = [sum(a * b for a, b in zip(query, v)) / math.sqrt(dim) for v in vectors]
                    top = max(scores)
                    result[layer, row, head, block] = (top + math.log(sum(math.exp(x - top) for x in scores))
                                                       if method == 'exact_mass' else top)
    return result


@pytest.mark.parametrize('method', METHODS)
def test_all_scorers_match_independent_gqa_partial_block_oracle(method):
    generator = torch.Generator().manual_seed(260128)
    keys = torch.randn(2, 4, 32, 2, 7, generator=generator)
    queries = torch.randn(2, 5, 6, 7, generator=generator)
    counts = [32, 9, 1, 0]
    for block, count in enumerate(counts):
        keys[:, block, count:] = torch.nan
    features = build_block_features(keys, counts, method)
    actual = block_logits(queries, features)
    expected = scalar_logits(keys, counts, queries, method)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.isneginf(actual[..., -1]).all()


def test_exact_mass_is_global_token_softmax_then_block_sum_with_eligible_mask():
    generator = torch.Generator().manual_seed(105)
    keys = torch.randn(2, 4, 7, 2, 5, generator=generator)
    queries = torch.randn(2, 11, 4, 5, generator=generator)
    counts, eligible = [7, 3, 5, 0], [True, False, True, True]
    features = build_block_features(keys, counts, 'exact_mass')
    logits = block_logits(queries, features)
    probabilities = logits.masked_fill(~torch.tensor(eligible), -torch.inf).softmax(-1)
    expected = torch.zeros_like(probabilities)
    for layer in range(2):
        for row in range(11):
            for head in range(4):
                operands = [(block, keys[layer, block, t, head // 2])
                            for block, count in enumerate(counts) if eligible[block] for t in range(count)]
                token_logits = torch.tensor([
                    sum(float(a) * float(b) for a, b in zip(queries[layer, row, head], key)) / math.sqrt(5)
                    for _, key in operands], dtype=torch.float64)
                token_probs = token_logits.softmax(0)
                for (block, _), probability in zip(operands, token_probs):
                    expected[layer, row, head, block] += probability
    torch.testing.assert_close(probabilities, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))


@pytest.mark.parametrize('method', METHODS)
def test_block_tiles_and_query_tiles_preserve_full_domain_scores(method):
    generator = torch.Generator().manual_seed(64)
    keys = torch.randn(2, 5, 32, 2, 4, generator=generator)
    queries = torch.randn(2, 19, 4, 4, generator=generator)
    counts = [32, 17, 31, 4, 1]
    whole = build_block_features(keys, counts, method)
    parts = [build_block_features(keys[:, :2], counts[:2], method),
             build_block_features(keys[:, 2:], counts[2:], method)]
    joined = concat_block_features(parts)
    torch.testing.assert_close(joined.values, whole.values, rtol=0, atol=0)
    torch.testing.assert_close(joined.valid, whole.valid, rtol=0, atol=0)
    expected = block_logits(queries, whole).softmax(-1).sum((0, 1, 2)) / (2 * 4)
    actual = torch.zeros(5)
    for chunk in queries.split(3, dim=1):
        logits = torch.cat([block_logits(chunk, part) for part in parts], dim=-1)
        actual += logits.softmax(-1).sum((0, 1, 2)) / (2 * 4)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-6)
    reordered = joined.select_blocks([4, 1, 0]).to('cpu')
    torch.testing.assert_close(block_logits(queries, reordered),
                               block_logits(queries, whole)[..., [4, 1, 0]])


def test_min_max_is_an_upper_bound_on_every_true_token_dot_product():
    generator = torch.Generator().manual_seed(56)
    keys = torch.randn(2, 3, 32, 2, 8, generator=generator)
    queries = torch.randn(2, 13, 6, 8, generator=generator)
    counts = [32, 7, 1]
    upper = block_logits(queries, build_block_features(keys, counts, 'min_max'))
    for block, count in enumerate(counts):
        for token in range(count):
            raw = keys[:, block, token].repeat_interleave(3, dim=1)
            exact = (queries * raw[:, None]).sum(-1) / math.sqrt(8)
            assert (upper[..., block] >= exact - 2e-6).all()


def test_fixed4_is_synthetic_cluster_means_and_has_four_slots_per_physical_block():
    keys = torch.zeros(1, 1, 64, 1, 2)
    keys[0, 0, :, 0] = torch.tensor([[1., 0.], [3., 0.], [0., 1.], [0., 3.],
                                    [-1., 0.], [-3., 0.], [0., -1.], [0., -3.]] * 8)
    fixed = build_block_features(keys, [64], 'fixed4')
    fps = build_block_features(keys, [64], 'fps4')
    assert fixed.values.shape == fps.values.shape == (1, 1, 4, 1, 2)
    expected = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    torch.testing.assert_close(fixed.values[0, 0, :, 0], expected, rtol=0, atol=0)
    for prototype in fixed.values[0, 0, :, 0]:
        assert not any(torch.equal(prototype, key) for key in keys[0, 0, :, 0])
    for representative in fps.values[0, 0, :, 0]:
        assert any(torch.equal(representative, key) for key in keys[0, 0, :, 0])


def test_fixed4_empty_groups_duplicate_first_seed_and_zero_blocks_remain_masked():
    keys = torch.full((1, 3, 32, 1, 2), torch.nan)
    keys[0, 0, :3, 0] = torch.tensor([[2., 0.], [4., 0.], [8., 0.]])
    keys[0, 1, :2, 0] = 0.
    features = build_block_features(keys, [3, 2, 0], 'fixed4')
    torch.testing.assert_close(features.values[0, 0, :, 0],
                               torch.tensor([[14. / 3, 0.], [2., 0.], [2., 0.], [2., 0.]]))
    assert features.valid[0, :2].all()
    assert not features.valid[0, 2].any()
    assert (features.values[0, 1:] == 0).all()


def test_fps_partial_slots_masked_and_ties_choose_earliest_real_token():
    keys = torch.zeros(1, 2, 32, 1, 2)
    keys[0, 0, :4, 0] = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    features = build_block_features(keys, [4, 2], 'fps4')
    torch.testing.assert_close(features.values[0, 0], keys[0, 0, :4], rtol=0, atol=0)
    assert features.valid[0, 1, :, 0].tolist() == [True, True, False, False]


@pytest.mark.parametrize('method', METHODS)
def test_fp8_inputs_embed_exactly_in_fp32_features(method):
    generator = torch.Generator().manual_seed(99)
    keys = torch.randint(-5, 6, (1, 2, 32, 1, 4), generator=generator).to(torch.float8_e4m3fn)
    actual = build_block_features(keys, [32, 5], method)
    expected = build_block_features(keys.float(), [32, 5], method)
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(actual.valid, expected.valid, rtol=0, atol=0)


@pytest.mark.parametrize('counts', [[-1], [33], [2.5], [True], [[2]], [2, 3]])
def test_invalid_token_counts_fail_closed(counts):
    with pytest.raises(ValueError, match='counts'):
        build_block_features(torch.zeros(1, 1, 32, 1, 4), counts, 'mean_k')


@pytest.mark.parametrize('method', METHODS)
def test_valid_nonfinite_keys_and_queries_fail_closed(method):
    keys = torch.zeros(1, 1, 32, 1, 4)
    keys[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite valid Key'):
        build_block_features(keys, [1], method)
    keys.zero_()
    features = build_block_features(keys, [1], method)
    with pytest.raises(ValueError, match='Nonfinite Query'):
        block_logits(torch.full((1, 1, 1, 4), torch.inf), features)


def test_unknown_method_and_incompatible_gqa_tiles_fail_closed():
    keys = torch.zeros(1, 1, 32, 2, 4)
    with pytest.raises(ValueError, match='Unknown recall method'):
        build_block_features(keys, [32], 'attention_fallback')
    mean = build_block_features(keys, [32], 'mean_k')
    with pytest.raises(ValueError, match='geometry'):
        block_logits(torch.zeros(1, 1, 3, 4), mean)
    with pytest.raises(ValueError, match='incompatible'):
        concat_block_features([mean, build_block_features(keys, [32], 'fps4')])


@pytest.mark.parametrize('method', METHODS)
def test_zero_query_has_uniform_block_mass_except_exact_mass_counts(method):
    keys = torch.zeros(1, 3, 32, 1, 2)
    features = build_block_features(keys, [32, 8, 0], method)
    probabilities = block_logits(torch.zeros(1, 1, 1, 2), features).softmax(-1).flatten()
    expected = torch.tensor([.8, .2, 0.]) if method == 'exact_mass' else torch.tensor([.5, .5, 0.])
    torch.testing.assert_close(probabilities, expected)
