// Flash Next's n-gram rows from the checkpoint: a batch of preads on a persistent pool of threads, without the GIL.
#include <torch/extension.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class Pool {
 public:
  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  // Each row of ``reads`` (fd, offset, size, at): ``size`` bytes at ``offset`` of ``fd`` into ``out + at``.
  void run(const int64_t* reads, int64_t n, uint8_t* out, int64_t threads) {
    std::lock_guard<std::mutex> call(call_);          // one batch at a time
    int64_t helpers = std::max<int64_t>(0, std::min<int64_t>(threads, n) - 1);
    {
      std::unique_lock<std::mutex> lock(m_);
      while (static_cast<int64_t>(threads_.size()) < helpers) threads_.emplace_back([this] { loop(); });
      reads_ = reads;
      n_ = n;
      out_ = out;
      next_.store(0);
      error_.clear();
      busy_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    wake_.notify_all();
    work();
    std::unique_lock<std::mutex> lock(m_);
    done_.wait(lock, [this] { return busy_ == 0; });
    if (!error_.empty()) throw std::runtime_error(error_);
  }

 private:
  void loop() {
    uint64_t seen = 0;
    {
      std::lock_guard<std::mutex> lock(m_);
      seen = generation_;           // a thread started for a batch joins it below
      --seen;
    }
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m_);
        wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      work();
      std::lock_guard<std::mutex> lock(m_);
      if (--busy_ == 0) done_.notify_all();
    }
  }

  void work() {
    for (;;) {
      int64_t i = next_.fetch_add(1);
      if (i >= n_) return;
      const int64_t* r = reads_ + 4 * i;
      int fd = static_cast<int>(r[0]);
      int64_t offset = r[1], size = r[2], done = 0;
      uint8_t* dst = out_ + r[3];
      while (done < size) {
        ssize_t got = pread(fd, dst + done, static_cast<size_t>(size - done), offset + done);
        const int err = errno;          // before anything else (the lock below) can change it
        if (got < 0 && err == EINTR) continue;
        if (got <= 0) {
          std::lock_guard<std::mutex> lock(m_);
          if (error_.empty()) {
            error_ = got == 0 ? "short read of the n-gram tables at byte " + std::to_string(offset + done) +
                                    ": the checkpoint changed"
                              : std::string("reading the n-gram tables: ") + std::strerror(err);
          }
          next_.store(n_);            // the rest of the batch is abandoned
          return;
        }
        done += got;
      }
    }
  }

  std::mutex call_, m_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> threads_;
  const int64_t* reads_ = nullptr;
  int64_t n_ = 0;
  uint8_t* out_ = nullptr;
  std::atomic<int64_t> next_{0};
  std::string error_;
  int busy_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

Pool& pool() {
  static Pool* p = new Pool();      // never destroyed: its threads may outlive interpreter teardown
  return *p;
}

void read_rows(torch::Tensor reads, torch::Tensor out, int64_t threads) {
  TORCH_CHECK(reads.device().is_cpu() && reads.scalar_type() == torch::kInt64 && reads.dim() == 2 &&
                  reads.size(1) == 4 && reads.is_contiguous(),
              "reads must be a contiguous CPU int64 [n, 4] tensor");
  TORCH_CHECK(out.device().is_cpu() && out.scalar_type() == torch::kUInt8 && out.is_contiguous(),
              "out must be a contiguous CPU uint8 tensor");
  const int64_t n = reads.size(0), cap = out.numel();
  const int64_t* r = reads.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(r[4 * i + 2] >= 0 && r[4 * i + 3] >= 0 && r[4 * i + 3] + r[4 * i + 2] <= cap && r[4 * i + 1] >= 0,
                "a read lands outside the output");
  }
  if (n == 0) return;
  pybind11::gil_scoped_release nogil;
  pool().run(r, n, out.data_ptr<uint8_t>(), std::max<int64_t>(1, threads));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_rows", &read_rows, "preads (fd, offset, size, at) into out, on up to `threads` threads, GIL released");
}
