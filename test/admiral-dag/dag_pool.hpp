#pragma once

// Wave-A DAG census node pool (math-clarity campaign D4c): the runtime store behind the
// symbolic V of sym_dag.hpp. One node per dynamically issued plane op; leaves are plane
// inputs (plane/layout/index plus a provenance tag that reporting reads and equality
// never does) and constants (memcpy'd T bits). A post-pass tagger turns cplain into cturn
// by bit-matching against the per-radix ct_sincos_turns fold set, attaching the exact
// (Num, Den, Conj) turn. Two equality levels, never conflated: L1 identity is the exact
// turn where tagged (rounded bits otherwise), L2 is always the rounded bits.
//
// The pool never rewrites ids. The intern table is a census-only hash-cons whose
// duplicate count is the report's CSE ratio; DAG equality goes through the canonizer, a
// memoized (op, canon-kids, payload) renumbering that two traced DAGs share so their
// canon ids are directly comparable. Direction normalization is a canon view: plane
// relabel for the kernel-boundary re/im swap, constant conjugation for dialects that
// fold direction into the twiddles.
//
// Test-local (test/ only): no shipped TU includes this header, so wave A moves zero
// shipped objects. C++17/C++20 dual-valid: no concepts, no consteval, no templated
// lambdas, no class-type NTTPs, no std::bit_cast, no spans.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

namespace admiral_dag {

enum class op : std::uint8_t {
    input,   // plane leaf: (plane, layout, index); load provenance recorded for the registry
    cplain,  // plain constant: rounded T bits, no exact identity attached
    cturn,   // constant the tagger matched to an exact (Num, Den, Conj) turn
    add,
    sub,
    neg,
    mul,
    fma,
    fnma,
    fms,
    swz,   // granule dialect unary shuffle net; reserved until the granule step
    shfl,  // granule dialect two-source shuffle; reserved
    flip   // granule dialect sign flip through an integer mask; reserved
};

inline constexpr std::size_t kOpCount = static_cast<std::size_t>(op::flip) + 1;

constexpr bool is_fp_op(op o) noexcept {
    switch (o) {
        case op::add:
        case op::sub:
        case op::neg:
        case op::mul:
        case op::fma:
        case op::fnma:
        case op::fms: return true;
        case op::input:
        case op::cplain:
        case op::cturn:
        case op::swz:
        case op::shfl:
        case op::flip: return false;
    }
    return false;  // unreachable: the switch is exhaustive over op
}

constexpr unsigned op_arity(op o) noexcept {
    switch (o) {
        case op::input:
        case op::cplain:
        case op::cturn: return 0;
        case op::neg:
        case op::swz:
        case op::flip: return 1;
        case op::add:
        case op::sub:
        case op::mul:
        case op::shfl: return 2;
        case op::fma:
        case op::fnma:
        case op::fms: return 3;
    }
    return 0;  // unreachable: the switch is exhaustive over op
}

constexpr const char* op_name(op o) noexcept {
    switch (o) {
        case op::input: return "input";
        case op::cplain: return "cplain";
        case op::cturn: return "cturn";
        case op::add: return "add";
        case op::sub: return "sub";
        case op::neg: return "neg";
        case op::mul: return "mul";
        case op::fma: return "fma";
        case op::fnma: return "fnma";
        case op::fms: return "fms";
        case op::swz: return "swz";
        case op::shfl: return "shfl";
        case op::flip: return "flip";
    }
    return "?";  // unreachable: the switch is exhaustive over op
}

enum class plane : std::uint8_t { re, im };
enum class layout : std::uint8_t { soa, aos };

// How the harness built the leaf: direct plane arrays (the caller handed whole aligned
// batches), load_unaligned, or masked load. Registry/report only: canonical identity
// excludes it, which is what makes the three kernel_batched provenance arms comparable.
enum class provenance : std::uint8_t { aligned, unaligned, masked, lift };

constexpr const char* provenance_name(provenance p) noexcept {
    switch (p) {
        case provenance::aligned: return "aligned";
        case provenance::unaligned: return "unaligned";
        case provenance::masked: return "masked";
        case provenance::lift: return "lift";
    }
    return "?";  // unreachable: the switch is exhaustive over provenance
}

struct node {
    op o = op::input;
    std::uint32_t kids[3] = {0u, 0u, 0u};
    plane pl = plane::re;                     // input: plane
    layout ly = layout::soa;                  // input: layout tag
    provenance prov = provenance::aligned;    // input: census-only load provenance
    std::uint32_t index = 0;                  // input: element index in its plane
    std::uint64_t bits = 0;                   // cplain/cturn: memcpy'd T bits, zero-extended
    std::uint32_t tnum = 0;                   // cturn: exact-turns tag, ct_sincos_turns order
    std::uint32_t tden = 0;
    bool tconj = false;
    // Granule dialect fields.
    std::uint32_t aux = 0;  // cplain: nonzero marks a non-uniform lane pool; bits is the
                            // fnv1a64 hash of all W lanes (exact lanes kept in pool::aux_)
    bool int_view = false;  // flip: the sym_int integer-domain marker; casts are zero-node views
};

template <typename T>
std::uint64_t bits_of(T v) noexcept {
    static_assert(sizeof(T) <= sizeof(std::uint64_t));
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof(T));
    return b;
}

inline std::uint64_t fnv1a64(const void* data, std::size_t n) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Hash of a compile-time index pattern (a shuffle net's batch_constant), salted so
// different net domains cannot collide.
template <typename Vt>
std::uint64_t net_hash(std::initializer_list<Vt> idx, std::uint64_t salt) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    const auto feed = [&h](const void* data, std::size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
    };
    feed(&salt, sizeof(salt));
    feed(idx.begin(), idx.size() * sizeof(Vt));
    return h;
}

// One fold-set row the tagger matches a cplain's bits against. Built test-side from
// ct_sincos_turns instantiations, in canonical scan order (den asc, num asc, conj false
// before true), so tagging is deterministic across traces that share the den list.
struct turn_entry {
    std::uint32_t num;
    std::uint32_t den;
    bool conj;
    std::uint64_t cbits;
    std::uint64_t sbits;
};

// Leaf descriptor the pointer registry hands to load* spellings.
struct input_slot {
    plane pl;
    layout ly;
    std::uint32_t index;
    provenance prov;
};

template <typename T>
using input_map = std::map<const T*, input_slot>;

constexpr std::uint64_t pack_input(plane pl, layout ly, std::uint32_t index) noexcept {
    return static_cast<std::uint64_t>(pl) | (static_cast<std::uint64_t>(ly) << 2)
           | (static_cast<std::uint64_t>(index) << 8);
}

// Canonical exact-value identity of a tagged constant: conjugation normalizes into num
// (ct_sincos_turns's own num -> den - num rule), so (Num, Den, true) and (Den-Num, Den,
// false) are one payload. The mark bit keeps a packed tag disjoint from any raw bit pair.
inline constexpr std::uint64_t kTurnMark = std::uint64_t{1} << 63;

constexpr std::uint64_t pack_turn(std::uint32_t num, std::uint32_t den, bool conj) noexcept {
    std::uint32_t n = num % den;
    if (conj && n != 0) n = den - n;
    return kTurnMark | static_cast<std::uint64_t>(n) | (static_cast<std::uint64_t>(den) << 21);
}

struct intern_key {
    std::uint8_t o;
    std::uint32_t k0, k1, k2;
    std::uint64_t payload;
    bool operator<(const intern_key& r) const noexcept {
        return std::tie(o, k0, k1, k2, payload) < std::tie(r.o, r.k0, r.k1, r.k2, r.payload);
    }
};

template <typename T>
class pool {
public:
    pool() = default;
    pool(const pool&) = delete;
    pool& operator=(const pool&) = delete;

    std::uint32_t make_input(plane pl, layout ly, std::uint32_t index, provenance prov) {
        node n;
        n.o = op::input;
        n.pl = pl;
        n.ly = ly;
        n.prov = prov;
        n.index = index;
        return append(n);
    }
    std::uint32_t make_const(T v) {
        node n;
        n.o = op::cplain;
        n.bits = bits_of(v);
        return append(n);
    }
    std::uint32_t make_turn(T v, std::uint32_t num, std::uint32_t den, bool conj) {
        node n;
        n.o = op::cturn;
        n.bits = bits_of(v);
        n.tnum = num;
        n.tden = den;
        n.tconj = conj;
        return append(n);
    }
    // Constant load covering W lanes. A uniform pool collapses to the broadcast constant;
    // anything else (the granule {-s,+s} pools) keeps one cplain with a full-lane hash.
    std::uint32_t make_const_lanes(const T* lanes, std::size_t w) {
        bool uniform = true;
        const std::uint64_t b0 = bits_of(lanes[0]);
        for (std::size_t i = 1; i < w; ++i)
            if (bits_of(lanes[i]) != b0) {
                uniform = false;
                break;
            }
        if (uniform) return make_const(lanes[0]);
        node n;
        n.o = op::cplain;
        n.bits = fnv1a64(lanes, w * sizeof(T));
        n.aux = static_cast<std::uint32_t>(aux_.size() + 1);
        for (std::size_t i = 0; i < w; ++i) aux_.push_back(bits_of(lanes[i]));
        return append(n);
    }
    std::uint32_t make_net1(op o, std::uint32_t a, std::uint64_t payload) {
        node n;
        n.o = o;
        n.kids[0] = a;
        n.bits = payload;
        return append(n);
    }
    std::uint32_t make_net2(op o, std::uint32_t a, std::uint32_t b, std::uint64_t payload) {
        node n;
        n.o = o;
        n.kids[0] = a;
        n.kids[1] = b;
        n.bits = payload;
        return append(n);
    }
    std::uint32_t make_flip(std::uint32_t a, std::uint64_t mask_hash) {
        const std::uint32_t id = make_net1(op::flip, a, mask_hash);
        nodes_[id].int_view = true;
        return id;
    }
    std::uint32_t make1(op o, std::uint32_t a) {
        node n;
        n.o = o;
        n.kids[0] = a;
        return append(n);
    }
    std::uint32_t make2(op o, std::uint32_t a, std::uint32_t b) {
        node n;
        n.o = o;
        n.kids[0] = a;
        n.kids[1] = b;
        return append(n);
    }
    std::uint32_t make3(op o, std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        node n;
        n.o = o;
        n.kids[0] = a;
        n.kids[1] = b;
        n.kids[2] = c;
        return append(n);
    }

    // Attaches (Num, Den, Conj) to every cplain whose bits occur in the fold set.
    // Returns the tagged count.
    std::size_t tag_turns(const std::vector<turn_entry>& set) {
        std::size_t tagged = 0;
        for (node& n : nodes_) {
            if (n.o != op::cplain) continue;
            for (const turn_entry& e : set) {
                if (n.bits == e.cbits || n.bits == e.sbits) {
                    n.o = op::cturn;
                    n.tnum = e.num;
                    n.tden = e.den;
                    n.tconj = e.conj;
                    ++tagged;
                    break;
                }
            }
        }
        return tagged;
    }

    const std::vector<node>& nodes() const noexcept { return nodes_; }
    std::size_t size() const noexcept { return nodes_.size(); }
    const std::array<std::uint64_t, kOpCount>& op_hist() const noexcept { return hist_; }
    std::uint64_t op_count() const noexcept {
        std::uint64_t t = 0;
        for (std::size_t i = 0; i < kOpCount; ++i)
            if (is_fp_op(static_cast<op>(i))) t += hist_[i];
        return t;
    }
    std::uint64_t created() const noexcept { return created_; }
    std::uint64_t intern_dups() const noexcept { return intern_dups_; }

private:
    std::uint32_t append(const node& n) {
        const std::uint32_t id = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back(n);
        hist_[static_cast<std::size_t>(n.o)] += 1;
        ++created_;
        const intern_key k{static_cast<std::uint8_t>(n.o), n.kids[0], n.kids[1], n.kids[2],
                           census_payload(n)};
        if (!intern_.emplace(k, id).second) ++intern_dups_;
        return id;
    }
    static std::uint64_t census_payload(const node& n) noexcept {
        if (n.o == op::input) return pack_input(n.pl, n.ly, n.index);
        return n.bits;  // cplain/cturn bits and net payloads; add..fms carry none (0 bits)
    }

    std::vector<node> nodes_;
    std::vector<std::uint64_t> aux_;  // exact lanes behind aux-marked cplain nodes
    std::array<std::uint64_t, kOpCount> hist_{};
    std::map<intern_key, std::uint32_t> intern_;
    std::uint64_t created_ = 0;
    std::uint64_t intern_dups_ = 0;
};

// Canonicalization knobs for direction normalization: swap_planes for the
// kernel-boundary re/im exchange, conjugate for dialects that fold direction into the
// twiddle constants.
struct canon_view {
    bool swap_planes = false;
    bool conjugate = false;
};

enum class level : std::uint8_t { l1, l2 };

struct canon_key {
    std::uint8_t o = 0;
    std::uint32_t k0 = 0;
    std::uint32_t k1 = 0;
    std::uint32_t k2 = 0;
    std::uint64_t payload = 0;
    bool operator<(const canon_key& r) const noexcept {
        return std::tie(o, k0, k1, k2, payload) < std::tie(r.o, r.k0, r.k1, r.k2, r.payload);
    }
};

// Memoized (op, canon-kids, payload) renumbering. canon ids are invariant node
// identities, so ids assigned canonizing DAG A stay equal to the ids the same
// substructure gets when DAG B is canonized into the same table afterwards.
class canonizer {
public:
    template <typename T>
    std::vector<std::uint32_t> run(const pool<T>& src, level lev, const canon_view& view) {
        std::vector<std::uint32_t> out(src.size(), 0u);
        const std::vector<node>& nd = src.nodes();
        for (std::uint32_t id = 0; id < nd.size(); ++id) {
            const node& n = nd[id];
            canon_key k;
            k.o = static_cast<std::uint8_t>(n.o);
            switch (n.o) {
                case op::input: {
                    const plane p =
                        view.swap_planes ? (n.pl == plane::re ? plane::im : plane::re) : n.pl;
                    k.payload = pack_input(p, n.ly, n.index);
                    break;
                }
                case op::cplain:
                    k.payload = n.bits;
                    break;
                case op::cturn:
                    k.payload = lev == level::l1
                                    ? pack_turn(n.tnum, n.tden, n.tconj != view.conjugate)
                                    : n.bits;
                    break;
                case op::add:
                case op::sub:
                case op::neg:
                case op::mul:
                case op::fma:
                case op::fnma:
                case op::fms:
                case op::swz:
                case op::shfl:
                case op::flip: {
                    // Kid slots past the op's arity hold raw 0s: remap only live slots.
                    // Remapping a dead slot through out[0] would key every op on the canon
                    // id of whatever node happened to be created first in its pool.
                    const unsigned ar = op_arity(n.o);
                    if (ar >= 1) k.k0 = out[n.kids[0]];
                    if (ar >= 2) k.k1 = out[n.kids[1]];
                    if (ar >= 3) k.k2 = out[n.kids[2]];
                    if (n.o == op::swz || n.o == op::shfl || n.o == op::flip)
                        k.payload = n.bits;  // shuffle-net / flip-mask pattern
                    break;
                }
            }
            const auto r = table_.emplace(k, next_);
            if (r.second) ++next_;
            out[id] = r.first->second;
        }
        return out;
    }
    std::size_t distinct() const noexcept { return table_.size(); }

private:
    std::map<canon_key, std::uint32_t> table_;
    std::uint32_t next_ = 0;
};

// One traced execution. family/arm identify the body and its if-constexpr arm, trips how
// many times the rung ran; width and prec describe the V the trace went through.
struct rung {
    const char* family;
    const char* arm;
    provenance prov;
    char prec;
    std::uint32_t radix;
    std::uint32_t width;
    std::uint32_t trips;
    std::uint64_t nodes;
};

inline std::vector<rung>& registry() {
    static std::vector<rung> rows;
    return rows;
}

}  // namespace admiral_dag
