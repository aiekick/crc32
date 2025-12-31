#include <cstdint> // uint32_t, uint8_t
#include <iostream>
#include <chrono>
#include <nmmintrin.h> // _mm_crc32_u8/u32/u64

#if defined(_MSC_VER)
#include <intrin.h>

static bool cpu_has_sse42() {
    int cpuInfo[4]{};
    __cpuid(cpuInfo, 1);
    // ECX bit 20 = SSE4.2
    return (cpuInfo[2] & (1 << 20)) != 0;
}
#endif

class Crc32 {
private:
    bool m_has_sse42{};
    uint32_t m_computed_crc32_lut[256]{};

    void m_compute_crc32_lut_table() {
        uint32_t crc32 = 1;
        for (unsigned int i = 128; i; i >>= 1) {
            crc32 = (crc32 >> 1) ^ (crc32 & 1 ? 0xedb88320 : 0);
            for (unsigned int j = 0; j < 256; j += 2 * i)
                m_computed_crc32_lut[i + j] = crc32 ^ m_computed_crc32_lut[j];
        }
    }

    uint32_t m_compute_crc32(uint32_t a_crc32, const void *ap_datas, size_t a_bytes_size) {
        if (m_computed_crc32_lut[255] == 0) { m_compute_crc32_lut_table(); }
        const uint8_t *p_bytes = static_cast<const uint8_t *>(ap_datas);
        for (size_t i = 0; i < a_bytes_size; ++i) {
            a_crc32 ^= p_bytes[i];
            a_crc32 = (a_crc32 >> 8) ^ m_computed_crc32_lut[a_crc32 & 0xFF];
        }
        return a_crc32;
    }

    uint32_t m_compute_crc32_sse4(uint32_t a_crc32,
                                  const void *ap_datas,
                                  size_t a_bytes_size) {
        const uint8_t *p8 = static_cast<const uint8_t *>(ap_datas);

#if defined(_M_X64) || defined(__x86_64__)
        while (a_bytes_size >= 8) {
            a_crc32 = static_cast<uint32_t>(
                _mm_crc32_u64(a_crc32,
                              *reinterpret_cast<const uint64_t *>(p8))
            );
            p8 += 8;
            a_bytes_size -= 8;
        }
#endif

        while (a_bytes_size--) {
            a_crc32 = _mm_crc32_u8(a_crc32, *p8++);
        }

        return a_crc32;
    }

    uint32_t m_crc32_value{};

public:
    Crc32() {
        m_has_sse42 = false;
#ifdef _MSC_VER
        m_has_sse42 = cpu_has_sse42();
#endif
#ifdef __SSE4_2__
        m_has_sse42 = true;
#endif
    }

    Crc32 &reset(uint32_t a_value = 0U) {
        m_crc32_value = a_value;
        return *this;
    }

    Crc32 &seed(uint32_t a_seed) {
        m_crc32_value = a_seed;
        return *this;
    }

    template<typename TTYPE>
    Crc32 &crc32(TTYPE a_value) {
        const void *data = reinterpret_cast<const void *>(&a_value);
        const size_t size = sizeof(TTYPE);
        if (m_has_sse42) {
            m_crc32_value = m_compute_crc32_sse4(m_crc32_value, data, size);
        } else {
            m_crc32_value = m_compute_crc32(m_crc32_value, data, size);
        }
        return *this;
    }

    uint32_t get() {
        return m_crc32_value ^ 0xFFFFFFFFu;
    }
};

int main() {
    using clock = std::chrono::steady_clock;
    struct Vec2 {
        float x{2.5f}, y{0.1f};
    } vec2;
    bool change{true};
    {
        volatile uint32_t sink = 0;
        constexpr int N = 1'000'000;
        const auto start = clock::now();
        for (int i = 0; i < N; ++i) {
            sink ^= Crc32().seed(125).crc32(vec2.x).crc32(change).get();
        }
        const auto end = clock::now();
        auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "avg ns = " << (ns / N) << "\n";
    }
    {
        volatile uint32_t sink = 0;
        constexpr int N = 1'000'000;
        const auto start = clock::now();
        for (int i = 0; i < N; ++i) {
            sink ^= Crc32().seed(125).crc32(vec2.x).crc32(vec2.y).crc32(change).get();
        }
        const auto end = clock::now();
        auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "avg ns = " << (ns / N) << "\n";
    }
}
