#pragma once
// Python 3 random.Random (MT19937) — Zobrist tables and shuffle must match.
#include <cstdint>
#include <vector>
#include <algorithm>

namespace gkt {

class PythonRandom {
public:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER_MASK = 0x80000000u;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffu;

    explicit PythonRandom(uint64_t seed = 0) { this->seed(seed); }

    void seed(uint64_t value) {
        uint32_t key[4];
        int nkey = 0;
        if (value == 0) {
            key[0] = 0;
            nkey = 1;
        } else {
            while (value && nkey < 4) {
                key[nkey++] = static_cast<uint32_t>(value & 0xffffffffu);
                value >>= 32;
            }
        }
        init_by_array(key, nkey);
    }

    uint32_t genrand_int32() {
        uint32_t y;
        static const uint32_t mag01[2] = {0x0u, MATRIX_A};
        if (index_ >= N) {
            int kk;
            for (kk = 0; kk < N - M; kk++) {
                y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            for (; kk < N - 1; kk++) {
                y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
            }
            y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    uint64_t getrandbits64() {
        uint32_t lo = genrand_int32();
        uint32_t hi = genrand_int32();
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }

    double random() {
        uint32_t a = genrand_int32() >> 5;
        uint32_t b = genrand_int32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    int randbelow(int n) {
        if (n <= 1) return 0;
        int k = 0;
        uint32_t m = static_cast<uint32_t>(n);
        while (m) { m >>= 1; k++; }
        uint32_t r;
        do {
            r = getrandbits(k);
        } while (r >= static_cast<uint32_t>(n));
        return static_cast<int>(r);
    }

    uint32_t getrandbits(int k) {
        if (k <= 0) return 0;
        if (k <= 32) return genrand_int32() >> (32 - k);
        return static_cast<uint32_t>(getrandbits64());
    }

    int randrange(int n) { return randbelow(n); }

    template <typename T>
    void shuffle(std::vector<T>& x) {
        for (int i = static_cast<int>(x.size()) - 1; i >= 1; --i) {
            int j = randbelow(i + 1);
            std::swap(x[i], x[j]);
        }
    }

private:
    uint32_t mt_[N]{};
    int index_ = N;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int mti = 1; mti < N; mti++) {
            mt_[mti] = 1812433253u * (mt_[mti - 1] ^ (mt_[mti - 1] >> 30)) + mti;
        }
        index_ = N;
    }

    void init_by_array(const uint32_t* init_key, int key_length) {
        init_genrand(19650218u);
        int i = 1, j = 0;
        int k = (N > key_length ? N : key_length);
        for (; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u))
                     + init_key[j] + static_cast<uint32_t>(j);
            i++; j++;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
            if (j >= key_length) j = 0;
        }
        for (k = N - 1; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u))
                     - static_cast<uint32_t>(i);
            i++;
            if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
        }
        mt_[0] = 0x80000000u;
        index_ = N;
    }
};

}  // namespace gkt
