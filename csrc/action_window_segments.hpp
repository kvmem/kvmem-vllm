// Read-only Action metadata adapter. The original window_segments algorithm
// and logical page identities are unchanged; no per-page Python tuple is made.
#pragma once

#include "qw3/kvmem_store.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kvmem_action {

struct WindowSegments {
    std::vector<int32_t> starts, counts;
    std::vector<uint64_t> history;
};

// The caller retains the request's producer role. This read-only operation
// holds the GIL throughout; it neither changes residency nor borrows pointers
// into Store beyond the call. The Python arrays returned by the binding own
// their storage and survive subsequent append/selection/Store destruction.
inline WindowSegments window_segments(const qw3::KvMemStore &store,
        int64_t history_begin, int64_t local_begin, int64_t count,
        int64_t block_tokens) {
    if (history_begin < 0 || local_begin < 0 || count < 0 || block_tokens <= 0)
        throw std::invalid_argument("Invalid action window coordinates");
    if (local_begin > std::numeric_limits<int32_t>::max() ||
            count > std::numeric_limits<int32_t>::max() - local_begin)
        throw std::invalid_argument("Action window exceeds int32 kernel coordinates");

    struct Record {
        uint64_t begin;
        uint32_t count;
        int64_t position;
    };
    std::vector<Record> records;
    for (const auto &block : store.blocks()) {
        if (block.repository_generation && block.n_tokens &&
                (block.in_working_set || block.tier == qw3::KvTier::GPU))
            records.push_back({block.history_pos_start, block.n_tokens, block.baked_pos});
    }
    std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return a.position < b.position;
    });
    WindowSegments result;
    int64_t cursor = 0;
    for (const auto &record : records) {
        if (record.position >= local_begin) break;
        // Check the gap before subtracting a potentially invalid baked_pos.
        // The old helper rejects the same row without publishing any result.
        if (record.position != cursor)
            throw std::invalid_argument("Action window has a gap or stale dense mapping");
        const int64_t n = std::min<int64_t>(record.count, local_begin-record.position);
        if (n <= 0 || n > block_tokens)
            throw std::invalid_argument("Action window has a gap or stale dense mapping");
        result.starts.push_back(static_cast<int32_t>(cursor));
        result.counts.push_back(static_cast<int32_t>(n));
        result.history.push_back(record.begin-record.begin%static_cast<uint64_t>(block_tokens));
        cursor += n;
    }
    // cursor <= local_begin, so this subtraction cannot overflow int64.
    int64_t logical = history_begin-(local_begin-cursor);
    if (logical < 0)
        throw std::invalid_argument("Invalid action suffix origin");
    const int64_t stop = local_begin+count;
    while (cursor < stop) {
        const int64_t n = std::min(block_tokens-logical%block_tokens, stop-cursor);
        result.starts.push_back(static_cast<int32_t>(cursor));
        result.counts.push_back(static_cast<int32_t>(n));
        result.history.push_back(static_cast<uint64_t>(logical-logical%block_tokens));
        cursor += n;
        // The final segment does not consume its exclusive logical end. Match
        // Python at INT64_MAX without rejecting an unused next coordinate.
        if (cursor < stop) {
            if (logical > std::numeric_limits<int64_t>::max()-n)
                throw std::invalid_argument("Action history exceeds int64 coordinates");
            logical += n;
        }
    }
    return result;
}

// True iff canonical_segments(window_segments(...), local_begin+count, block_tokens)
// (serving/native_attention_planning.py) equals the frozen decode frame's rows
// truncated at that extent, exactly as NativeAttentionStats.resolve builds its
// expected prefix (rows starting before the extent, the last count clipped).
// Any invalid window or frame answers false: the caller then runs the original
// Python path, which raises or records the same reason as before.
inline bool window_matches_frame(const qw3::KvMemStore &store, int64_t history_begin,
        int64_t local_begin, int64_t count, int64_t block_tokens,
        const int64_t *frame_starts, const int64_t *frame_counts, const int64_t *frame_history,
        size_t frame_rows) {
    WindowSegments segments;
    try {
        segments = window_segments(store, history_begin, local_begin, count, block_tokens);
    } catch (const std::invalid_argument &) {
        return false;
    }
    const int64_t length = local_begin+count;
    // _canonical: rows that start before the extent, last count clipped,
    // contiguous from zero to the extent, adjacent equal keys merged.
    std::vector<int64_t> run_begin, run_key;
    int64_t cursor = 0;
    for (size_t i = 0; i < segments.starts.size(); ++i) {
        const int64_t start = segments.starts[i];
        if (start >= length) continue;
        const int64_t n = std::min<int64_t>(segments.counts[i], length-start);
        if (n <= 0 || start != cursor) return false;
        const int64_t key = static_cast<int64_t>(segments.history[i]);
        if (run_key.empty() || run_key.back() != key) {
            run_begin.push_back(start);
            run_key.push_back(key);
        }
        cursor = start+n;
    }
    if (run_begin.empty() || cursor != length) return false;
    // canonical_segments: split runs longer than block_tokens into block-sized
    // chunks (shorter runs are one chunk), then compare row by row.
    size_t row = 0;
    for (size_t r = 0; r < run_begin.size(); ++r) {
        const int64_t begin = run_begin[r];
        const int64_t total = (r+1 < run_begin.size() ? run_begin[r+1] : length)-begin;
        for (int64_t offset = 0; offset < total; offset += block_tokens, ++row) {
            const int64_t start = begin+offset, n = std::min(block_tokens, total-offset);
            if (row >= frame_rows || frame_starts[row] != start ||
                    std::min<int64_t>(frame_counts[row], length-frame_starts[row]) != n ||
                    frame_history[row] != run_key[r])
                return false;
        }
    }
    // The frame must have no further row starting before the extent.
    return row == frame_rows || frame_starts[row] >= length;
}

} // namespace kvmem_action
