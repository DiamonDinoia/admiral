#pragma once

// ORDER CONTRACT: this header must follow every xsimd include and PRECEDE every admiral
// include in the translation unit. The kernel text reaches fma/fnma/fms through QUALIFIED
// xsimd:: calls whose lookup binds at each template's definition point, so a shim declared
// after an admiral header is invisible there (probe negative control:
// /tmp/adm-dag-probe/probe_order_neg.cpp hard-errors out of simd_swizzle.hpp). This
// header includes xsimd itself, then dag_pool, then declares the namespace-xsimd shims;
// the including TU puts every admiral include after it.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include <xsimd/xsimd.hpp>

#include "dag_pool.hpp"

namespace admiral_dag {

// One live context per (T, trace): the pool the ops append to, the opt-in pointer
// registry the load* spellings resolve (plane, index) from, and the fallback counter the
// harness REQUIREs zero when every load should have resolved. read_through sends an
// unmapped load to the pointed-at lane values (the granule dialect's constant pools, real
// rodata memory); lift_next indexes the real-batch lift leaves; sunk counts values
// downcast out of sym land.
template <typename T>
struct sym_ctx {
    pool<T>* p = nullptr;
    const input_map<T>* m = nullptr;
    bool read_through = false;
    std::uint64_t unmapped_loads = 0;
    std::uint32_t lift_next = 0;
    std::uint64_t sunk = 0;
};

// Symbolic V handle: (pool, node id) plus an uninit marker. The default ctor is the
// marker every kernel array starts from; any value read of a marker bumps a counter the
// tests REQUIRE zero. One sym serves flyweight duty for the SoA dialect's batch-the-tabs
// width (size = xsimd::batch<T>::size at the real default arch, never a fake one).
template <typename T>
class sym {
public:
    using value_type = T;
    using arch_type = xsimd::default_arch;
    static constexpr std::size_t size = xsimd::batch<T>::size;

    sym() noexcept = default;
    sym(T v) : p_(ambient_pool()), id_(p_->make_const(v)), init_(true) {}
    // Scalar-tail seam: a dif_butterfly<T, R, T> / apply_stage_twiddle<T, ..., T>
    // instantiated at T = sym<float> still spells the shipped static_cast<T>(w.c) on a
    // double fold. The exact-match overload applies the shipped rounding explicitly,
    // where the implicit double -> float hop through the converting ctor would trip
    // -Wfloat-conversion.
    template <typename U = T, typename std::enable_if<std::is_same_v<U, float>, int>::type = 0>
    sym(double v) : sym(static_cast<T>(v)) {}
    // Lift from the real-batch seam (the granule dialect's f32 row-tile transpose returns
    // real batches): a leaf, not a computation.
    sym(xsimd::batch<T, xsimd::default_arch>)
        : sym(input_lift()) {}

    sym(const sym&) noexcept = default;
    sym& operator=(const sym&) noexcept = default;

    sym operator+(sym o) const { return make2(op::add, *this, o); }
    sym operator-(sym o) const { return make2(op::sub, *this, o); }
    sym operator*(sym o) const { return make2(op::mul, *this, o); }
    sym operator-() const { return make1(op::neg, *this); }

    // The load/store surface of the V contract. Loads resolve (plane, index) through the
    // context's pointer registry; an unresolved pointer lands in a sentinel leaf and bumps
    // ctx.unmapped_loads. Stores are read checks only.
    static sym load_aligned(const T* ptr) { return load_via_map(ptr, provenance::aligned); }
    static sym load_unaligned(const T* ptr) { return load_via_map(ptr, provenance::unaligned); }
    template <typename Mask>
    static sym load(const T* ptr, const Mask&, xsimd::unaligned_mode) {
        return load_via_map(ptr, provenance::masked);
    }
    void store_aligned(T*) const { read_check_self(); }
    void store_unaligned(T*) const { read_check_self(); }
    template <typename Mask>
    void store(T*, const Mask&, xsimd::unaligned_mode) const {
        read_check_self();
    }

    // Harness surface.
    static sym input(plane pl, layout ly, std::uint32_t index, provenance prov) {
        pool<T>* p = ambient_pool();
        return sym(p->make_input(pl, ly, index, prov), p);
    }

    static void bind(sym_ctx<T>* c) { ambient() = c; }
    static sym_ctx<T>* current() { return ambient(); }
    static sym_ctx<T>& ctx() {
        sym_ctx<T>* c = ambient();
        if (c == nullptr || c->p == nullptr) std::abort();  // unbound trace is a harness bug
        return *c;
    }

    static std::uint64_t uninit_reads() { return uninit_counter(); }
    static void reset_counters() { uninit_counter() = 0; }

    std::uint32_t id() const { return id_; }

    // Node makers, one fused node per call: the census counts as shipped and never
    // re-decides contraction. Public because the namespace-xsimd shims call them.
    static sym make1(op o, sym a) {
        a.read_check_self();
        pool<T>* p = a.p_ != nullptr ? a.p_ : ambient_pool();
        return sym(p->make1(o, a.id_), p);
    }
    static sym make2(op o, sym a, sym b) {
        read_check_pair(a, b);
        pool<T>* p = a.p_ != nullptr ? a.p_ : (b.p_ != nullptr ? b.p_ : ambient_pool());
        return sym(p->make2(o, a.id_, b.id_), p);
    }
    static sym make3(op o, sym a, sym b, sym c) {
        read_check_pair(a, b);
        c.read_check_self();
        pool<T>* p = a.p_ != nullptr ? a.p_
                                     : (b.p_ != nullptr ? b.p_
                                                        : (c.p_ != nullptr ? c.p_
                                                                           : ambient_pool()));
        return sym(p->make3(o, a.id_, b.id_, c.id_), p);
    }

    // Granule dialect surface: shuffle nets and the int-view flip. A payload hash names
    // the constant net inside the node; DAG equality compares (op, kids, payload).
    static sym make_swz(op o, sym a, std::uint64_t net) {
        a.read_check_self();
        pool<T>* p = a.p_ != nullptr ? a.p_ : ambient_pool();
        return sym(p->make_net1(o, a.id_, net), p);
    }
    static sym make_shfl(op o, sym a, sym b, std::uint64_t net) {
        read_check_pair(a, b);
        pool<T>* p = a.p_ != nullptr ? a.p_ : (b.p_ != nullptr ? b.p_ : ambient_pool());
        return sym(p->make_net2(o, a.id_, b.id_, net), p);
    }
    // View attachment for the int-view / bitcast shims: no node, just the handle.
    static sym attach(pool<T>* p, std::uint32_t id) { return sym(id, p); }
    static pool<T>* pool_of(sym a) { return a.p_; }
    static std::uint32_t id_of(sym a) { return a.id_; }

private:
    sym(std::uint32_t id, pool<T>* p) noexcept : p_(p), id_(id), init_(true) {}

    static sym_ctx<T>*& ambient() {
        static sym_ctx<T>* c = nullptr;
        return c;
    }
    static pool<T>* ambient_pool() { return ctx().p; }
    static std::uint64_t& uninit_counter() {
        static std::uint64_t u = 0;
        return u;
    }

    void read_check_self() const {
        if (!init_) ++uninit_counter();
    }
    static void read_check_pair(const sym& a, const sym& b) {
        if (!a.init_ || !b.init_) ++uninit_counter();
    }

    static sym load_via_map(const T* ptr, provenance prov) {
        sym_ctx<T>& c = ctx();
        if (c.m != nullptr) {
            const auto it = c.m->find(ptr);
            if (it != c.m->end()) {
                const input_slot& s = it->second;
                return sym(c.p->make_input(s.pl, s.ly, s.index, prov), c.p);
            }
        }
        if (c.read_through)  // granule constant pools: the pointed-at lanes ARE the value
            return sym(c.p->make_const_lanes(ptr, size), c.p);
        ++c.unmapped_loads;
        return sym(c.p->make_input(plane::re, layout::soa, kUnmappedIndex, prov), c.p);
    }

    static sym input_lift() {
        sym_ctx<T>& c = ctx();
        return sym(c.p->make_input(plane::re, layout::aos, c.lift_next++, provenance::lift),
                   c.p);
    }

    static constexpr std::uint32_t kUnmappedIndex = 0xFFFFFFFFu;

    pool<T>* p_ = nullptr;
    std::uint32_t id_ = 0;
    bool init_ = false;
};

// In-namespace companion of sym for the granule dialect's integer domain. Carries one
// shared node; only make_flip appends (with the integer-domain marker), because the
// bitwise_cast views themselves are layout moves, zero nodes by construction.
template <typename U, typename F>
class sym_int {
public:
    using unsigned_type = U;
    using float_type = F;
    sym_int() noexcept = default;
    sym_int(admiral_dag::pool<F>* p, std::uint32_t id) noexcept : p_(p), id_(id) {}

    sym_int make_flip(std::uint64_t mask_hash) const {
        return sym_int(p_, p_->make_flip(id_, mask_hash));
    }

    admiral_dag::pool<F>* p_ = nullptr;
    std::uint32_t id_ = 0;
};

}  // namespace admiral_dag

// The qualified-call shims the kernel text binds; see the ORDER CONTRACT at the top of
// this header for why they must sit here and nowhere else.
namespace xsimd {

template <typename T>
admiral_dag::sym<T> fma(admiral_dag::sym<T> a, admiral_dag::sym<T> b, admiral_dag::sym<T> c) {
    return admiral_dag::sym<T>::make3(admiral_dag::op::fma, a, b, c);
}
template <typename T>
admiral_dag::sym<T> fnma(admiral_dag::sym<T> a, admiral_dag::sym<T> b, admiral_dag::sym<T> c) {
    return admiral_dag::sym<T>::make3(admiral_dag::op::fnma, a, b, c);
}
template <typename T>
admiral_dag::sym<T> fms(admiral_dag::sym<T> a, admiral_dag::sym<T> b, admiral_dag::sym<T> c) {
    return admiral_dag::sym<T>::make3(admiral_dag::op::fms, a, b, c);
}

// The free-function spellings the granule bodies use.
template <typename T>
admiral_dag::sym<T> add(admiral_dag::sym<T> a, admiral_dag::sym<T> b) {
    return a + b;
}
template <typename T>
admiral_dag::sym<T> sub(admiral_dag::sym<T> a, admiral_dag::sym<T> b) {
    return a - b;
}
template <typename T>
admiral_dag::sym<T> mul(admiral_dag::sym<T> a, admiral_dag::sym<T> b) {
    return a * b;
}
template <typename T>
admiral_dag::sym<T> neg(admiral_dag::sym<T> a) {
    return -a;
}

template <typename T, typename Vt, typename A, Vt... Vs>
admiral_dag::sym<T> swizzle(admiral_dag::sym<T> a, batch_constant<Vt, A, Vs...>) {
    const std::uint64_t net = admiral_dag::net_hash<Vt>({Vs...}, sizeof(Vt));
    return admiral_dag::sym<T>::make_swz(admiral_dag::op::swz, a, net);
}
template <typename T, typename Vt, typename A, Vt... Vs>
admiral_dag::sym<T> shuffle(admiral_dag::sym<T> a, admiral_dag::sym<T> b,
                            batch_constant<Vt, A, Vs...>) {
    const std::uint64_t net = admiral_dag::net_hash<Vt>({Vs...}, sizeof(Vt));
    return admiral_dag::sym<T>::make_shfl(admiral_dag::op::shfl, a, b, net);
}

// Whole-block transpose, reached at the double-granularity 8x8 form. Coarse model: one
// two-source node per output register; the net's exact stage wiring is invisible to the
// census and irrelevant to its counts.
template <typename T>
void transpose(admiral_dag::sym<T>* first, admiral_dag::sym<T>* last) {
    const std::size_t n = static_cast<std::size_t>(last - first);
    const std::vector<admiral_dag::sym<T>> src(first, last);
    for (std::size_t j = 0; j < n; ++j) {
        const std::uint64_t net = admiral_dag::net_hash<std::size_t>({j, n}, 0x74726e73ull);
        first[j] = admiral_dag::sym<T>::make_shfl(admiral_dag::op::shfl, src[j],
                                                  src[(j + n / 2) % n], net);
    }
}

// Cast views. Same-width: an int-view handle over the same node. Cross-width (the f32
// tile's out-cast to double): a real-batch sink; the sym side just counts the downcast.
template <typename To, typename T, typename std::enable_if<sizeof(To) == sizeof(T), int>::type = 0>
admiral_dag::sym_int<To, T> bitwise_cast(const admiral_dag::sym<T>& v) {
    return admiral_dag::sym_int<To, T>(admiral_dag::sym<T>::pool_of(v),
                                       admiral_dag::sym<T>::id_of(v));
}
template <typename To, typename T, typename std::enable_if<sizeof(To) != sizeof(T), int>::type = 0>
batch<To, typename admiral_dag::sym<T>::arch_type> bitwise_cast(const admiral_dag::sym<T>&) {
    using S = admiral_dag::sym<T>;
    ++S::ctx().sunk;
    return batch<To, typename S::arch_type>(To(0));
}
template <typename To, typename U, typename F>
admiral_dag::sym<To> bitwise_cast(const admiral_dag::sym_int<U, F>& v) {
    static_assert(sizeof(To) == sizeof(F), "int view casts back at its own granularity");
    return admiral_dag::sym<To>::attach(v.p_, v.id_);
}

// The flip: one integer-domain node, the mask's lane pattern hashed into its payload.
template <typename U, typename A, typename F>
admiral_dag::sym_int<U, F> bitwise_xor(admiral_dag::sym_int<U, F> v, batch<U, A> mask) {
    constexpr std::size_t W = batch<U, A>::size;
    std::array<U, W> lanes;
    mask.store_unaligned(lanes.data());
    const std::uint64_t h = admiral_dag::fnv1a64(lanes.data(), W * sizeof(U));
    return v.make_flip(h);
}

}  // namespace xsimd
