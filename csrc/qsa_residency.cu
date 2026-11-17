// QSA streaming residency (Flash Next, P2). One QSA layer per launch.
//
// plan:      one CTA per batch request. Collects the distinct 4-token groups the
//            request's rows selected outside the ring, keeps resident groups,
//            gives each missing group a hot slot (free, then not recently used,
//            then any unpinned slot; a sweep clears reference bits when the
//            clear slots run out), appends (slot, group, hot slot) to the copy
//            list, and rewrites every selected position to its hot-buffer
//            position: hot groups at j*4 + (pos & 3) in pages [0, hot_pages),
//            ring tokens at hot_tokens + pos % ring_tokens.
// copy:      persistent CTAs copy listed groups (K and V planes, 4 tokens each)
//            from the pinned host pool into the layer's hot pages over UVA.
// writeback: copies the K/V the step just wrote into the ring back to the pool,
//            so the pool holds every frame token before its ring slot is reused.
//
// Attention reads the same tokens in the same column order, so outputs do not
// depend on which hot slot a group occupies. Requests whose state slot is not
// streaming are left untouched.
//
// Invariant kept by the caller: the ring page holds the newest ring_tokens
// positions of the request's frame (seeded from the pool at every refresh,
// then kept by the step's own K/V writes), and the pool holds every position
// older than that.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int kThreads = 512;
constexpr int kGroup = 4;

struct Stats {            // device counters, read without a per-step sync
  unsigned long long needed;
  unsigned long long misses;
  unsigned long long steps;
  unsigned long long errors;
};

__device__ int block_exclusive_scan(int value, int* scratch, int* total) {
  // Hillis-Steele over kThreads values in shared memory; deterministic.
  int t = threadIdx.x;
  scratch[t] = value;
  __syncthreads();
  for (int offset = 1; offset < kThreads; offset <<= 1) {
    int add = t >= offset ? scratch[t - offset] : 0;
    __syncthreads();
    scratch[t] += add;
    __syncthreads();
  }
  int inclusive = scratch[t];
  if (t == kThreads - 1) *total = inclusive;
  __syncthreads();
  return inclusive - value;
}

__global__ void plan_kernel(
    int32_t* __restrict__ selected, int64_t row_stride, int width,
    const int32_t* __restrict__ query_start, const int64_t* __restrict__ query_positions,
    const int32_t* __restrict__ request_slot, const uint8_t* __restrict__ streaming,
    int32_t* __restrict__ group_slot, int64_t max_groups,
    int32_t* __restrict__ slot_group, uint8_t* __restrict__ referenced, int hot_groups,
    int32_t* __restrict__ copy_list, int32_t* __restrict__ copy_count,
    int hot_tokens, int ring_tokens, int ring_margin, Stats* __restrict__ stats) {
  extern __shared__ uint32_t shared[];
  int request = blockIdx.x;
  int first = query_start[request], last = query_start[request + 1];
  if (last <= first) return;
  int slot = request_slot[request];
  if (slot < 0 || !streaming[slot]) return;
  int words = (int)((max_groups + 31) / 32);
  int hot_words = (hot_groups + 31) / 32;
  uint32_t* needed = shared;                       // [words]
  uint32_t* pinned = needed + words;               // [hot_words]
  int* misses = reinterpret_cast<int*>(pinned + hot_words);   // [hot_groups]
  int* victims = misses + hot_groups;              // [hot_groups]
  int* scan = victims + hot_groups;                // [kThreads]
  int* totals = scan + kThreads;                   // [4]
  int32_t* groups = group_slot + (int64_t)slot * max_groups;
  int32_t* owners = slot_group + (int64_t)slot * hot_groups;
  uint8_t* refs = referenced + (int64_t)slot * hot_groups;

  for (int i = threadIdx.x; i < words; i += kThreads) needed[i] = 0;
  for (int i = threadIdx.x; i < hot_words; i += kThreads) pinned[i] = 0;
  __syncthreads();
  // Positions within ring_margin of the newest possible write may already be
  // overwritten by speculative writes past this step; they come from the pool.
  int64_t ring_start = query_positions[last - 1] - ring_tokens + 1 + ring_margin;
  // 1. distinct groups outside the ring (causal entries only: attention masks the rest)
  for (int row = first; row < last; ++row) {
    const int32_t* entries = selected + row * row_stride;
    int64_t limit = query_positions[row];
    for (int c = threadIdx.x; c < width; c += kThreads) {
      int32_t pos = entries[c];
      if (pos >= 0 && pos < ring_start && pos <= limit) {
        int64_t g = pos / kGroup;
        if (g < max_groups) atomicOr(&needed[g >> 5], 1u << (g & 31));
        else atomicAdd(&stats->errors, 1ull);
      }
    }
  }
  __syncthreads();
  // 2. hits pin their slot; misses are listed in group order
  int my_misses = 0, my_needed = 0;
  for (int w = threadIdx.x; w < words; w += kThreads) {
    uint32_t bits = needed[w];
    while (bits) {
      int g = w * 32 + __ffs(bits) - 1;
      bits &= bits - 1;
      ++my_needed;
      int j = groups[g];
      if (j >= 0) {
        atomicOr(&pinned[j >> 5], 1u << (j & 31));
        refs[j] = 1;
      } else {
        ++my_misses;
      }
    }
  }
  int miss_base = block_exclusive_scan(my_misses, scan, &totals[0]);
  int miss_total = totals[0];
  for (int w = threadIdx.x, k = miss_base; w < words; w += kThreads) {
    uint32_t bits = needed[w];
    while (bits) {
      int g = w * 32 + __ffs(bits) - 1;
      bits &= bits - 1;
      if (groups[g] < 0 && k < hot_groups) misses[k++] = g;
    }
  }
  block_exclusive_scan(my_needed, scan, &totals[1]);
  int needed_total = totals[1];
  __syncthreads();
  // 3. victims: unpinned slots, free first, then clear reference bit, then marked
  int counts[3] = {0, 0, 0};
  for (int j = threadIdx.x; j < hot_groups; j += kThreads) {
    if (pinned[j >> 5] & (1u << (j & 31))) continue;
    counts[owners[j] < 0 ? 0 : (refs[j] ? 2 : 1)]++;
  }
  int base[3];
  base[0] = block_exclusive_scan(counts[0], scan, &totals[0]);
  int free_total = totals[0];
  base[1] = block_exclusive_scan(counts[1], scan, &totals[1]);
  int clear_total = totals[1];
  base[2] = block_exclusive_scan(counts[2], scan, &totals[2]);
  int marked_total = totals[2];
  if (miss_total > free_total + clear_total + marked_total) {
    if (threadIdx.x == 0) atomicAdd(&stats->errors, 1ull);
    miss_total = free_total + clear_total + marked_total;
  }
  int offset[3] = {base[0], free_total + base[1], free_total + clear_total + base[2]};
  for (int j = threadIdx.x; j < hot_groups; j += kThreads) {
    if (pinned[j >> 5] & (1u << (j & 31))) continue;
    int category = owners[j] < 0 ? 0 : (refs[j] ? 2 : 1);
    int position = offset[category]++;
    if (position < miss_total) victims[position] = j;
  }
  __syncthreads();
  bool sweep = miss_total > free_total + clear_total;
  if (sweep) {
    for (int j = threadIdx.x; j < hot_groups; j += kThreads)
      if (!(pinned[j >> 5] & (1u << (j & 31)))) refs[j] = 0;
  }
  __syncthreads();
  // 4. evict and assign; one copy entry per miss
  for (int k = threadIdx.x; k < miss_total; k += kThreads) {
    int j = victims[k], g = misses[k];
    int old = owners[j];
    if (old >= 0) groups[old] = -1;
    groups[g] = j;
    owners[j] = g;
    refs[j] = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0 && miss_total) {
    int at = atomicAdd(copy_count, miss_total);
    totals[3] = at;
  }
  __syncthreads();
  for (int k = threadIdx.x; k < miss_total; k += kThreads) {
    int32_t* entry = copy_list + 3ll * (totals[3] + k);
    entry[0] = slot;
    entry[1] = misses[k];
    entry[2] = victims[k];
  }
  // 5. rewrite to hot-buffer positions. Attention then runs with query positions
  // that mask nothing, so the causal mask is applied here: QSA may select
  // positions after the row (later rows of the same verify interval).
  for (int row = first; row < last; ++row) {
    int32_t* entries = selected + row * row_stride;
    int64_t limit = query_positions[row];
    for (int c = threadIdx.x; c < width; c += kThreads) {
      int32_t pos = entries[c];
      if (pos < 0) continue;
      if (pos > limit) {
        entries[c] = -1;
      } else if (pos >= ring_start) {
        entries[c] = hot_tokens + (int32_t)(pos % ring_tokens);
      } else {
        int j = groups[pos / kGroup];
        if (j < 0) {
          atomicAdd(&stats->errors, 1ull);
          entries[c] = -1;
        } else {
          entries[c] = j * kGroup + (pos & (kGroup - 1));
        }
      }
    }
  }
  if (threadIdx.x == 0) {
    atomicAdd(&stats->steps, 1ull);
    atomicAdd(&stats->misses, (unsigned long long)miss_total);
    atomicAdd(&stats->needed, (unsigned long long)needed_total);
  }
}

// Copy listed groups host pool -> hot pages. A warp moves one plane (K or V) of
// one group: kGroup tokens x token_bytes, 16 bytes per lane per iteration.
__global__ void copy_kernel(const int32_t* __restrict__ copy_list, const int32_t* __restrict__ copy_count,
                            const int64_t* __restrict__ pool_base, int64_t pool_layer_offset, int64_t pool_plane_stride,
                            uint8_t* __restrict__ cache, int64_t page_stride, int64_t plane_stride,
                            const int32_t* __restrict__ hot_pages, int hot_page_columns,
                            int page_tokens, int token_bytes) {
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int warps = (gridDim.x * blockDim.x) >> 5;
  int count = *copy_count;
  int vectors = kGroup * token_bytes / 16;
  for (int item = warp; item < 2 * count; item += warps) {
    const int32_t* entry = copy_list + 3ll * (item >> 1);
    int plane = item & 1;
    int slot = entry[0], g = entry[1], j = entry[2];
    int token = j * kGroup;
    int page = hot_pages[slot * hot_page_columns + token / page_tokens];
    const int4* src = reinterpret_cast<const int4*>(
        reinterpret_cast<const uint8_t*>(pool_base[slot]) + pool_layer_offset + plane * pool_plane_stride +
        (int64_t)g * kGroup * token_bytes);
    int4* dst = reinterpret_cast<int4*>(cache + page * page_stride + plane * plane_stride +
                                        (int64_t)(token % page_tokens) * token_bytes);
    for (int v = lane; v < vectors; v += 32) dst[v] = src[v];
  }
}

// Ring -> pool for every row this step wrote (rows of streaming requests).
__global__ void writeback_kernel(const int32_t* __restrict__ row_request, const int64_t* __restrict__ query_positions,
                                 int rows, const int32_t* __restrict__ request_slot,
                                 const uint8_t* __restrict__ streaming,
                                 const int64_t* __restrict__ pool_base, int64_t pool_layer_offset,
                                 int64_t pool_plane_stride, const uint8_t* __restrict__ cache, int64_t page_stride,
                                 int64_t plane_stride, const int32_t* __restrict__ hot_pages, int hot_page_columns,
                                 int ring_column, int page_tokens, int token_bytes) {
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int warps = (gridDim.x * blockDim.x) >> 5;
  int vectors = token_bytes / 16;
  for (int item = warp; item < 2 * rows; item += warps) {
    int row = item >> 1, plane = item & 1;
    int request = row_request[row];
    if (request < 0) continue;
    int slot = request_slot[request];
    if (slot < 0 || !streaming[slot]) continue;
    int64_t f = query_positions[row];
    // Ring pages are the row's columns from ring_column on; frame column c writes to ring page c % r.
    int ring_pages = hot_page_columns - ring_column;
    int page = hot_pages[slot * hot_page_columns + ring_column + (int)((f / page_tokens) % ring_pages)];
    const int4* src = reinterpret_cast<const int4*>(cache + page * page_stride + plane * plane_stride +
                                                    (f % page_tokens) * token_bytes);
    int4* dst = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(pool_base[slot]) + pool_layer_offset +
                                        plane * pool_plane_stride + f * token_bytes);
    for (int v = lane; v < vectors; v += 32) dst[v] = src[v];
  }
}

// Staging fill, entirely on the device (no host read of the step's shape).
// The staging request is the streaming request with more rows than one verify
// interval; at most one per step (scheduler.py). Frame tokens [0, end) go to
// the staging run in page layout: positions below ring_start from the pinned
// pool, the newest ones from the ring pages (pool copies of those may lag).
__global__ void stage_fill_kernel(const int32_t* __restrict__ query_start, int requests,
                                  const int64_t* __restrict__ query_positions,
                                  const int32_t* __restrict__ request_slot, const uint8_t* __restrict__ streaming,
                                  int verify_rows, const int32_t* __restrict__ staging_pages, int staging_count,
                                  const int64_t* __restrict__ pool_base, int64_t pool_layer_offset,
                                  int64_t pool_plane_stride, uint8_t* __restrict__ cache, int64_t page_stride,
                                  int64_t plane_stride, const int32_t* __restrict__ hot_pages, int hot_page_columns,
                                  int ring_column, int page_tokens, int token_bytes, int ring_margin,
                                  int ring_only, Stats* __restrict__ stats) {
  __shared__ int chosen[2];   // request, count
  if (threadIdx.x == 0) {
    chosen[0] = -1; chosen[1] = 0;
    for (int r = 0; r < requests; ++r) {
      int first = query_start[r], last = query_start[r + 1];
      if (last - first <= verify_rows) continue;
      int slot = request_slot[r];
      if (slot < 0 || !streaming[slot]) continue;
      chosen[0] = r; chosen[1]++;
    }
    if (chosen[1] > 1 && blockIdx.x == 0) atomicAdd(&stats->errors, 1ull);
  }
  __syncthreads();
  int request = chosen[0];
  if (request < 0 || chosen[1] > 1) return;
  int slot = request_slot[request];
  int64_t end = query_positions[query_start[request + 1] - 1] + 1;
  int ring_pages = hot_page_columns - ring_column;
  int64_t ring_start = end - (int64_t)ring_pages * page_tokens + ring_margin;
  if (ring_start < 0) ring_start = 0;
  if (end > (int64_t)staging_count * page_tokens) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(&stats->errors, 1ull);
    return;
  }
  const int32_t* ring = hot_pages + (int64_t)slot * hot_page_columns + ring_column;
  const uint8_t* pool = reinterpret_cast<const uint8_t*>(pool_base[slot]) + pool_layer_offset;
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int warps = (gridDim.x * blockDim.x) >> 5;
  int vectors = token_bytes / 16;
  // ring_only: the pool part was prefetched on the copy stream; only the
  // ring window (written this step or recently) is copied on the critical path.
  int64_t begin = ring_only ? 2 * ring_start : 0;
  for (int64_t item = begin + warp; item < 2 * end; item += warps) {
    int64_t f = item >> 1;
    int plane = (int)(item & 1);
    const uint8_t* src = f < ring_start
        ? pool + plane * pool_plane_stride + f * token_bytes
        : cache + (int64_t)ring[(f / page_tokens) % ring_pages] * page_stride + plane * plane_stride +
              (f % page_tokens) * token_bytes;
    uint8_t* dst = cache + (int64_t)staging_pages[f / page_tokens] * page_stride + plane * plane_stride +
                   (f % page_tokens) * token_bytes;
    const int4* s4 = reinterpret_cast<const int4*>(src);
    int4* d4 = reinterpret_cast<int4*>(dst);
    for (int v = lane; v < vectors; v += 32) d4[v] = s4[v];
  }
}

}  // namespace

int64_t plan_shared_bytes(int64_t max_groups, int64_t hot_groups) {
  int64_t words = (max_groups + 31) / 32, hot_words = (hot_groups + 31) / 32;
  return 4 * (words + hot_words + 2 * hot_groups + kThreads + 4);
}

void residency_plan(torch::Tensor selected, int64_t width, torch::Tensor query_start, torch::Tensor query_positions,
                    torch::Tensor request_slot, torch::Tensor streaming, torch::Tensor group_slot,
                    torch::Tensor slot_group, torch::Tensor referenced, torch::Tensor copy_list,
                    torch::Tensor copy_count, int64_t requests, int64_t hot_tokens, int64_t ring_tokens,
                    int64_t ring_margin, torch::Tensor stats) {
  TORCH_CHECK(selected.dtype() == torch::kInt32 && group_slot.dtype() == torch::kInt32);
  TORCH_CHECK(query_positions.dtype() == torch::kInt64 && stats.numel() == 4);
  int64_t max_groups = group_slot.size(1), hot_groups = slot_group.size(1);
  auto stream = c10::cuda::getCurrentCUDAStream();
  int64_t shared = plan_shared_bytes(max_groups, hot_groups);
  TORCH_CHECK(shared <= 96 * 1024, "QSA residency plan exceeds shared memory");
  if (shared > 48 * 1024)
    cudaFuncSetAttribute(plan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
  if (requests <= 0) return;
  plan_kernel<<<requests, kThreads, shared, stream>>>(
      selected.data_ptr<int32_t>(), selected.stride(0), (int)width, query_start.data_ptr<int32_t>(),
      query_positions.data_ptr<int64_t>(), request_slot.data_ptr<int32_t>(), streaming.data_ptr<uint8_t>(),
      group_slot.data_ptr<int32_t>(), max_groups, slot_group.data_ptr<int32_t>(), referenced.data_ptr<uint8_t>(),
      (int)hot_groups, copy_list.data_ptr<int32_t>(), copy_count.data_ptr<int32_t>(), (int)hot_tokens,
      (int)ring_tokens, (int)ring_margin, reinterpret_cast<Stats*>(stats.data_ptr<int64_t>()));
}

void residency_copy(torch::Tensor copy_list, torch::Tensor copy_count, torch::Tensor pool_base,
                    int64_t pool_layer_offset, int64_t pool_plane_stride, torch::Tensor cache, int64_t page_stride,
                    int64_t plane_stride, torch::Tensor hot_pages, int64_t page_tokens, int64_t token_bytes,
                    int64_t blocks) {
  TORCH_CHECK(cache.dtype() == torch::kUInt8 && token_bytes % 16 == 0);
  copy_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      copy_list.data_ptr<int32_t>(), copy_count.data_ptr<int32_t>(), pool_base.data_ptr<int64_t>(),
      pool_layer_offset, pool_plane_stride, cache.data_ptr<uint8_t>(), page_stride, plane_stride,
      hot_pages.data_ptr<int32_t>(), (int)hot_pages.size(1), (int)page_tokens, (int)token_bytes);
}

void residency_writeback(torch::Tensor row_request, torch::Tensor query_positions, int64_t rows,
                         torch::Tensor request_slot, torch::Tensor streaming, torch::Tensor pool_base,
                         int64_t pool_layer_offset, int64_t pool_plane_stride, torch::Tensor cache,
                         int64_t page_stride, int64_t plane_stride, torch::Tensor hot_pages, int64_t ring_column,
                         int64_t page_tokens, int64_t token_bytes, int64_t blocks) {
  TORCH_CHECK(cache.dtype() == torch::kUInt8 && token_bytes % 16 == 0);
  if (rows <= 0) return;
  writeback_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      row_request.data_ptr<int32_t>(), query_positions.data_ptr<int64_t>(), (int)rows,
      request_slot.data_ptr<int32_t>(), streaming.data_ptr<uint8_t>(), pool_base.data_ptr<int64_t>(),
      pool_layer_offset, pool_plane_stride, cache.data_ptr<uint8_t>(), page_stride, plane_stride,
      hot_pages.data_ptr<int32_t>(), (int)hot_pages.size(1), (int)ring_column, (int)page_tokens, (int)token_bytes);
}

void stage_fill(torch::Tensor query_start, int64_t requests, torch::Tensor query_positions,
                torch::Tensor request_slot, torch::Tensor streaming, int64_t verify_rows,
                torch::Tensor staging_pages, torch::Tensor pool_base, int64_t pool_layer_offset,
                int64_t pool_plane_stride, torch::Tensor cache, int64_t page_stride, int64_t plane_stride,
                torch::Tensor hot_pages, int64_t ring_column, int64_t page_tokens, int64_t token_bytes,
                int64_t ring_margin, torch::Tensor stats, int64_t blocks, bool ring_only) {
  TORCH_CHECK(cache.dtype() == torch::kUInt8 && token_bytes % 16 == 0);
  TORCH_CHECK(query_start.dtype() == torch::kInt32 && request_slot.dtype() == torch::kInt32 &&
              staging_pages.dtype() == torch::kInt32 && query_positions.dtype() == torch::kInt64);
  if (requests <= 0) return;
  stage_fill_kernel<<<blocks, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      query_start.data_ptr<int32_t>(), (int)requests, query_positions.data_ptr<int64_t>(),
      request_slot.data_ptr<int32_t>(), streaming.data_ptr<uint8_t>(), (int)verify_rows,
      staging_pages.data_ptr<int32_t>(), (int)staging_pages.numel(), pool_base.data_ptr<int64_t>(),
      pool_layer_offset, pool_plane_stride, cache.data_ptr<uint8_t>(), page_stride, plane_stride,
      hot_pages.data_ptr<int32_t>(), (int)hot_pages.size(1), (int)ring_column, (int)page_tokens, (int)token_bytes,
      (int)ring_margin, (int)ring_only, reinterpret_cast<Stats*>(stats.data_ptr<int64_t>()));
}

// Strided copy on the current stream; UVA resolves host-pool and device pointers.
void memcpy2d(int64_t dst, int64_t dst_pitch, int64_t src, int64_t src_pitch, int64_t width, int64_t height) {
  if (width <= 0 || height <= 0) return;
  C10_CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void*>(dst), dst_pitch, reinterpret_cast<const void*>(src),
                                   src_pitch, width, height, cudaMemcpyDefault, c10::cuda::getCurrentCUDAStream()));
}

// Pin an existing host range (mapped, portable); returns its device address.
int64_t host_register(int64_t ptr, int64_t bytes) {
  void* host = reinterpret_cast<void*>(ptr);
  C10_CUDA_CHECK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* device = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
  return reinterpret_cast<int64_t>(device);
}

void host_unregister(int64_t ptr) {
  C10_CUDA_CHECK(cudaHostUnregister(reinterpret_cast<void*>(ptr)));
}

// A uint8 CUDA tensor over mapped host memory (no ownership), so existing
// kernels can read or write the pool by device address.
torch::Tensor device_view(int64_t device_ptr, int64_t bytes, int64_t device_index) {
  auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_index);
  return torch::from_blob(reinterpret_cast<void*>(device_ptr), {bytes}, [](void*) {}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("memcpy2d", &memcpy2d);
  m.def("stage_fill", &stage_fill);
  m.def("host_register", &host_register);
  m.def("host_unregister", &host_unregister);
  m.def("device_view", &device_view);
  m.def("plan_shared_bytes", &plan_shared_bytes);
  m.def("residency_plan", &residency_plan);
  m.def("residency_copy", &residency_copy);
  m.def("residency_writeback", &residency_writeback);
}
