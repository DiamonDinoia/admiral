#pragma once

// The seam between the public API (api.cpp, built once at the floor ISA) and one engine tier
// (cpp_api.hpp, built once per ISA tier). The public classes hold these bases; each tier
// derives its own states from them, so a plan runs the tier that built it.

#include "admiral/admiral.hpp"

#include <memory>

namespace admiral::detail {

template<typename T>
struct plan_state {
    virtual ~plan_state() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    virtual void run(bool is_forward, std::complex<T>* data, const T* fct) const = 0;
    virtual void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
                     const T* fct) const = 0;
};

template<typename T>
struct axis_state {
    virtual ~axis_state() = default;
    virtual void execute_bands(std::complex<T>* data, span<const std::size_t> lo,
                               span<const std::size_t> hi, std::size_t lo2_last,
                               std::size_t hi2_last, std::optional<T> fct) const = 0;
};

template<typename T>
struct strides_state {
    virtual ~strides_state() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    virtual void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
                     std::optional<T> fct) const = 0;
};

template<typename T>
struct real_state {
    virtual ~real_state() = default;
    [[nodiscard]] virtual std::size_t real_size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t cplx_size() const noexcept = 0;
    virtual void forward(const T* in, std::complex<T>* out, std::optional<T> fct) const = 0;
    virtual void inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const = 0;
};

template<typename T>
struct r2r_state {
    virtual ~r2r_state() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    virtual void forward(const T* in, T* out, std::optional<T> fct) const = 0;
    virtual void inverse(const T* in, T* out, std::optional<T> fct) const = 0;
};

// One tier's entry points. axis, strides and r2r stay null for long double.
template<typename T>
struct tier_ops {
    std::unique_ptr<plan_state<T>> (*plan)(span<const std::size_t>, const options&);
    std::unique_ptr<real_state<T>> (*real)(span<const std::size_t>, const options&);
    std::unique_ptr<axis_state<T>> (*axis)(span<const std::size_t>, std::size_t, bool,
                                           const options&);
    std::unique_ptr<strides_state<T>> (*strides)(std::size_t, std::size_t, std::size_t,
                                                 std::size_t, std::size_t, std::size_t,
                                                 const options&);
    std::unique_ptr<r2r_state<T>> (*r2r)(std::size_t, r2r_kind, std::size_t, const options&);
    void (*one_1d)(span<const std::complex<T>>, span<std::complex<T>>, bool, const options&,
                   std::optional<T>);
    void (*one_nd)(std::complex<T>*, span<const std::size_t>, bool, const options&,
                   std::optional<T>);
    void (*one_r2c)(const T*, std::complex<T>*, span<const std::size_t>, const options&,
                    std::optional<T>);
    void (*one_c2r)(std::complex<T>*, T*, span<const std::size_t>, const options&,
                    std::optional<T>);
};

// Called by the run-time dispatch mechanism of xsimd once per precision with the best available
// Arch. Each tier TU defines it for its own xsimd::default_arch only, so the Arch in the mangled
// name keeps the tiers apart.
struct tier_entry {
    template<class Arch, typename T>
    const tier_ops<T>* operator()(Arch, T*) const noexcept;
};

// The slot that api.cpp fills on the first call and every public entry reads. The tests
// overwrite it to pin a tier.
template<typename T>
const tier_ops<T>*& tier_slot();

}
