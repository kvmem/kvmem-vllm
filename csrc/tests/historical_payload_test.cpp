#include "historical_payload.hpp"
#include "qw3/kvmem_repository.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using Arena = vllm_kvmem::HistoricalPayloadArena;
using Repository = qw3::KvMemRepository;

static void check(bool value, const char *why) {
    if (!value) throw std::runtime_error(why);
}

template <typename Predicate>
static void eventually(Predicate predicate, const char *why) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        check(std::chrono::steady_clock::now() < end, why);
        std::this_thread::yield();
    }
}

static void independent_cpu_and_gpu_lifetimes() {
    Repository repo({1, 32});
    auto gpu = repo.try_reserve(0, 32).commit();
    Arena arena({1, 32, 1, 1, 1, false});
    const Arena::Handle h{gpu.handle().slot, gpu.handle().generation};
    std::array<uint8_t, 32> source{};
    source.fill(42);
    arena.publish(h, source.data(), source.size());
    auto reader = arena.acquire(h);
    auto dma = reader.clone();
    check(bool(reader) && reader.data()[0] == 42, "CPU payload publication failed");
    check(repo.retire_permanently(gpu.handle()) == Repository::RetireResult::Committed,
          "active GPU lease prevented logical retirement");
    check(arena.retire(h) == Arena::RetireResult::Committed, "CPU retirement failed");
    check(arena.retire(h) == Arena::RetireResult::AlreadyRetired, "CPU retirement not idempotent");
    check(!arena.acquire(h), "new CPU reader entered retired data");
    check(arena.advance() == 0 && arena.stats().used_bytes == 32, "CPU reader was freed early");
    reader.reset();
    check(arena.advance() == 0 && dma.data()[31] == 42, "in-flight DMA source was freed");
    dma.reset();
    check(arena.stats().released_slots == 0, "foreground last-reader drop freed payload");
    check(arena.advance() == 1 && arena.reclaimed(h), "ready CPU payload was not released");
    check(arena.stats().used_bytes == 0 && arena.stats().released_bytes == 32,
          "CPU bytes remained tied to GPU metadata references");
    check(gpu.record().token_begin == 0, "CPU release invalidated existing GPU operation");
    check(repo.ready_slots() == 0, "CPU release incorrectly returned the whole repository slot");
    check(repo.try_reclaim(gpu.handle(), [](const auto &) { return true; }) ==
              Repository::ReclaimResult::DeferredReaders,
          "repository slot was reused while GPU operation still owned it");
    const auto old = gpu.handle();
    gpu.reset();
    check(repo.try_reclaim(old, [&](const auto &) { return arena.reclaimed(h); }) ==
              Repository::ReclaimResult::Reclaimed,
          "metadata did not become reusable after all resource owners finished");
    eventually([&] { return arena.close(); }, "CPU arena shutdown did not finish");
}

static void generation_reuse_and_immutable_copy() {
    Arena arena({1, 16, 1, 1, 1, false});
    std::array<uint8_t, 16> source{};
    for (uint64_t generation = 1; generation <= 256; ++generation) {
        const Arena::Handle h{0, generation};
        source.fill(uint8_t(generation % 251));
        arena.publish(h, source.data(), source.size());
        source.fill(0);
        auto reader = arena.acquire(h);
        check(reader.data()[7] == uint8_t(generation % 251), "publication retained a mutable source");
        if (generation > 1) {
            check(arena.retire({0, generation - 1}) == Arena::RetireResult::StaleHandle,
                  "late retirement deleted a reused CPU allocation");
            check(!arena.acquire({0, generation - 1}), "old handle read replacement payload");
        }
        check(arena.retire(h) == Arena::RetireResult::Committed, "replacement retirement failed");
        reader.reset();
        check(arena.advance() == 1, "CPU allocation was not recycled");
        check(arena.stats().used_slots == 0, "bounded CPU directory leaked allocations");
    }
    check(arena.stats().released_slots == 256, "CPU reuse accounting mismatch");
    eventually([&] { return arena.close(); }, "CPU reuse test shutdown failed");
}

static void sparse_arena_reclamation_uses_compact_retirement_hints() {
    constexpr uint32_t capacity = 100000;
    Arena arena({capacity, 1, 1, 256, 64, false});
    const Arena::Handle h{capacity - 1, 1};
    const uint8_t byte = 7;
    arena.publish(h, &byte, 1);
    check(arena.retire(h) == Arena::RetireResult::Committed, "sparse retirement failed");
    // 100K records occupy 1563 bitmap words. Seven 256-word slices suffice;
    // scanning 256 individual records per slice would take 391 iterations.
    for (unsigned i = 0; i < 7 && !arena.reclaimed(h); ++i) arena.advance();
    check(arena.reclaimed(h), "sparse arena scanned every live/empty record before reclaiming");
    eventually([&] { return arena.close(); }, "sparse arena shutdown failed");
}

static void native_background_release_and_nonblocking_close() {
    Arena arena({2, 32, 1, 2, 2, true});
    std::array<uint8_t, 32> source{};
    arena.publish({0, 1}, source.data(), source.size());
    auto reader = arena.acquire({0, 1});
    check(!arena.close(), "shutdown freed an active CPU reader");
    check(!arena.acquire({0, 1}), "closing arena admitted a new reader");
    check(reader.data()[0] == 0, "closing invalidated an existing CPU reader");
    reader.reset();
    eventually([&] { return arena.close(); }, "native reclaimer waited for Python or GPU work");
    const auto stats = arena.stats();
    check(stats.used_slots == 0 && stats.released_slots == 1, "shutdown did not release CPU storage");
    check(stats.last_release_thread != std::hash<std::thread::id>{}(std::this_thread::get_id()),
          "payload destruction ran on the foreground thread");
}

static void concurrent_readers_cannot_observe_reused_payloads() {
    constexpr uint32_t capacity = 4;
    Arena arena({capacity, 32, 1, capacity, capacity, true});
    std::array<std::atomic<uint64_t>, capacity> generations{};
    std::atomic<bool> stop{false}, corrupt{false};
    std::atomic<uint64_t> observations{0};
    std::thread reader([&] {
        while (!stop.load(std::memory_order_acquire)) {
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                const auto generation = generations[slot].load(std::memory_order_acquire);
                if (!generation) continue;
                auto view = arena.acquire({slot, generation});
                if (!view) continue;
                const auto expected = uint8_t(generation % 251);
                for (size_t i = 0; i < view.size(); ++i)
                    if (view.data()[i] != expected) corrupt.store(true);
                ++observations;
            }
        }
    });
    struct JoinReader {
        std::atomic<bool> &stop;
        std::thread &thread;
        ~JoinReader() {
            stop.store(true, std::memory_order_release);
            if (thread.joinable()) thread.join();
        }
    } join{stop, reader};
    for (uint64_t generation = 1; generation <= 1000; ++generation) {
        const auto slot = uint32_t((generation - 1) % capacity);
        const auto prior = generations[slot].load();
        if (prior) eventually([&] { return arena.reclaimed({slot, prior}); }, "CPU reuse stalled");
        std::array<uint8_t, 32> source{};
        source.fill(uint8_t(generation % 251));
        arena.publish({slot, generation}, source.data(), source.size());
        generations[slot].store(generation, std::memory_order_release);
        if (generation == 1)
            eventually([&] { return observations.load() > 0; }, "reader did not enter published data");
        Arena::RetireResult retired = Arena::RetireResult::Retry;
        eventually([&] {
            retired = arena.retire({slot, generation});
            return retired != Arena::RetireResult::Retry;
        }, "bounded retirement retries did not make progress");
        check(retired == Arena::RetireResult::Committed, "concurrent CPU retirement failed");
    }
    stop.store(true, std::memory_order_release);
    reader.join();
    eventually([&] { return arena.close(); }, "concurrent CPU arena shutdown stalled");
    check(!corrupt && observations > 0, "reader observed a stale CPU allocation");
    check(arena.stats().released_slots == 1000 && arena.stats().readers == 0,
          "concurrent CPU arena leaked references or payloads");
}

static void external_storage_releases_only_on_native_worker() {
    Arena arena({1, 16, 1, 1, 1, true});
    struct State { std::atomic<unsigned> released{0}; std::atomic<size_t> thread{0}; } state;
    auto owner = arena.retain_external(&state, [](void *p) {
        auto &s = *static_cast<State *>(p);
        s.thread.store(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        s.released.fetch_add(1);
    }, 64);
    auto fence = owner->completion();
    owner->add_bytes(32);
    check(arena.stats().external_bytes == 96, "external index bytes not charged");
    auto second = arena.retain_external(&state, [](void *) {}, 16);
    check(arena.stats().external_owners == 2 && arena.stats().external_bytes == 112,
          "split native index ownership was not charged");
    bool full = false;
    try { arena.retain_external(&state, [](void *) {}, 1); }
    catch (const std::runtime_error &) { full = true; }
    check(full && state.released == 0, "capacity rejection consumed external ownership");
    check(!arena.close() && !fence->done.load(), "close ignored the live external owner");
    owner.reset();
    second.reset();
    eventually([&] { return fence->done.load(std::memory_order_acquire); }, "external release stalled");
    eventually([&] { return arena.close(); }, "external close stalled");
    check(state.released == 1 && state.thread != std::hash<std::thread::id>{}(std::this_thread::get_id()),
          "external storage destroyed on frontend or more than once");
    check(arena.stats().external_owners == 0 && arena.stats().external_bytes == 0 &&
          arena.stats().external_released_bytes == 112, "external storage accounting leaked");
}

static void external_completion_grants_immediately_reusable_capacity() {
    Arena arena({1,16,1,1,1,true});
    std::atomic<unsigned> released{0};
    for (unsigned i = 0; i < 200; ++i) {
        auto owner = arena.retain_external(&released, [](void *p) {
            static_cast<std::atomic<unsigned> *>(p)->fetch_add(1);
        },16);
        auto fence = owner->completion();
        owner.reset();
        eventually([&] { return fence->done.load(std::memory_order_acquire); }, "index fence stalled");
        check(arena.stats().external_owners == 0, "index fence preceded reusable owner capacity");
    }
    eventually([&] { return arena.close(); }, "index reuse shutdown stalled");
    check(released == 200, "index generation was lost or destroyed twice");
}

int main() {
    try {
        independent_cpu_and_gpu_lifetimes();
        generation_reuse_and_immutable_copy();
        sparse_arena_reclamation_uses_compact_retirement_hints();
        native_background_release_and_nonblocking_close();
        concurrent_readers_cannot_observe_reused_payloads();
        external_storage_releases_only_on_native_worker();
        external_completion_grants_immediately_reusable_capacity();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Historical CPU payload lifetime tests passed (1000 concurrent reuses)\n";
}
