/* Copyright 2026 WarpX contributors
 * License: BSD-3-Clause-LBNL
 */
#include "NativeYeeEnergyBalance.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Parallelization/WarpXSumGuardCells.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/ParticleBoundaries.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Reduce.H>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef WARPX_DIM_3D
namespace
{
    using Real = amrex::Real;
    using Pair = std::array<Real, 2>;
    using Fields = std::array<amrex::MultiFab const*, 3>;
    using Copies = std::array<std::unique_ptr<amrex::MultiFab>, 3>;
    using Arrays = amrex::GpuArray<amrex::Array4<Real const>, 3>;
    using Tuple = amrex::GpuTuple<Real, Real>;

    Fields View (Copies const& fields)
    {
        return {fields[0].get(), fields[1].get(), fields[2].get()};
    }

    Arrays GetArrays (Fields const& fields, amrex::MFIter const& mfi)
    {
        return {fields[0]->const_array(mfi), fields[1]->const_array(mfi),
                fields[2]->const_array(mfi)};
    }

    void Copy (Copies& dst, Fields const& src)
    {
        for (int d = 0; d < 3; ++d) {
            if (!dst[d]) {
                dst[d] = std::make_unique<amrex::MultiFab>(src[d]->boxArray(),
                    src[d]->DistributionMap(), 1, src[d]->nGrowVect());
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(dst[d]->boxArray() == src[d]->boxArray()
                && dst[d]->DistributionMap() == src[d]->DistributionMap(),
                "NativeYeeEnergyBalance does not support changing grid decompositions");
            amrex::MultiFab::Copy(*dst[d], *src[d], 0, 0, 1, src[d]->nGrowVect());
        }
    }

    // Only the diagnostic copy changes. In particular, physical guards are
    // copied from the actual stages, never extrapolated or filled by a BC.
    void Midpoint (Copies& old, Fields const& after)
    {
        for (int d = 0; d < 3; ++d) {
            amrex::MultiFab::LinComb(*old[d], Real(0.5), *old[d], 0,
                Real(0.5), *after[d], 0, 0, 1, old[d]->nGrowVect());
        }
    }

    template <typename Factory>
    Pair ReduceCells (amrex::MultiFab const& layout, Factory const& factory)
    {
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
        amrex::ReduceData<Real, Real> data(ops);
        // Each physical cell appears once. Native products use adjacent edges
        // or faces, so shared native nodes get trapezoidal weights automatically.
        for (amrex::MFIter mfi(layout); mfi.isValid(); ++mfi) {
            auto const box = amrex::enclosedCells(mfi.validbox());
            auto const function = factory(mfi);
            // Forward the device callable directly. An extended CUDA lambda
            // here would enclose the function-local Factory type in its
            // parent function's template arguments, which nvcc rejects.
            ops.eval(box, data, function);
        }
        auto const result = data.value();
        Pair out{amrex::get<0>(result), amrex::get<1>(result)};
        amrex::ParallelDescriptor::ReduceRealSum(out.data(), 2);
        return out;
    }

    Pair Dot (Fields const& left, Fields const& right, bool electric, Real factor)
    {
        auto result = ReduceCells(*left[0], [&](amrex::MFIter const& mfi)
        {
            auto const l = GetArrays(left, mfi);
            auto const r = GetArrays(right, mfi);
            return [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real sum = 0;
                Real absolute = 0;
                for (int d = 0; d < 3; ++d) {
                    int const t = (d + 1) % 3;
                    int const u = (d + 2) % 3;
                    int const n = electric ? 4 : 2;
                    for (int a = 0; a < n; ++a) {
                        amrex::GpuArray<int, 3> p{i, j, k};
                        if (electric) {
                            p[t] += a % 2;
                            p[u] += a / 2;
                        } else {
                            p[d] += a;
                        }
                        Real const product = l[d](p[0], p[1], p[2])
                            * r[d](p[0], p[1], p[2]) / Real(n);
                        sum += product;
                        absolute += std::abs(product);
                    }
                }
                return {sum, absolute};
            };
        });
        result[0] *= factor;
        result[1] *= std::abs(factor);
        return result;
    }

    // electric=true: <E,curl_backward B>; false: <B,curl_forward E>.
    Pair CurlDot (Fields const& dot, Fields const& source, bool electric,
                  amrex::GpuArray<Real, 3> const& inv_dx, Real factor)
    {
        auto result = ReduceCells(*dot[0], [&](amrex::MFIter const& mfi)
        {
            auto const v = GetArrays(dot, mfi);
            auto const s = GetArrays(source, mfi);
            return [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                Real sum = 0;
                Real absolute = 0;
                for (int d = 0; d < 3; ++d) {
                    int const t = (d + 1) % 3;
                    int const u = (d + 2) % 3;
                    int const n = electric ? 4 : 2;
                    for (int a = 0; a < n; ++a) {
                        amrex::GpuArray<int, 3> p{i, j, k};
                        if (electric) {
                            p[t] += a % 2;
                            p[u] += a / 2;
                        } else {
                            p[d] += a;
                        }
                        auto q = p;
                        auto r = p;
                        q[t] += electric ? -1 : 1;
                        r[u] += electric ? -1 : 1;
                        Real const orient = electric ? Real(-1) : Real(1);
                        Real const coefficient = orient * v[d](p[0], p[1], p[2]) / Real(n);
                        // Keep the absolute elementary-product scale before
                        // subtracting neighboring terms or curl components.
                        Real const a1 = coefficient * s[u](q[0], q[1], q[2]) * inv_dx[t];
                        Real const a2 = coefficient * s[u](p[0], p[1], p[2]) * inv_dx[t];
                        Real const b1 = coefficient * s[t](r[0], r[1], r[2]) * inv_dx[u];
                        Real const b2 = coefficient * s[t](p[0], p[1], p[2]) * inv_dx[u];
                        sum += a1 - a2 - b1 + b2;
                        absolute += std::abs(a1) + std::abs(a2) + std::abs(b1) + std::abs(b2);
                    }
                }
                return {sum, absolute};
            };
        });
        result[0] *= factor;
        result[1] *= std::abs(factor);
        return result;
    }

    Pair Face (Fields const& electric, Fields const& magnetic, amrex::Box const& domain,
               int normal, bool high, Real factor)
    {
        int const adjacent = high ? domain.bigEnd(normal) : domain.smallEnd(normal);
        int const face = adjacent + (high ? 1 : 0);
        auto result = ReduceCells(*electric[0], [&](amrex::MFIter const& mfi)
        {
            auto const e = GetArrays(electric, mfi);
            auto const b = GetArrays(magnetic, mfi);
            return [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> Tuple
            {
                amrex::GpuArray<int, 3> cell{i, j, k};
                if (cell[normal] != adjacent) { return {Real(0), Real(0)}; }
                int const t = (normal + 1) % 3;
                int const u = (normal + 2) % 3;
                Real sum = 0;
                Real absolute = 0;
                for (int term = 0; term < 2; ++term) {
                    int const et = term == 0 ? t : u;
                    int const bt = term == 0 ? u : t;
                    Real const sign = term == 0 ? Real(1) : Real(-1);
                    for (int edge = 0; edge < 2; ++edge) {
                        auto p = cell;
                        p[normal] = face;
                        p[bt] += edge;
                        // B is averaged across the two *normal* cell centers
                        // to this E edge, then multiplied at this same edge.
                        auto q = p;
                        --q[normal];
                        Real const ep = e[et](p[0], p[1], p[2]);
                        Real const a = Real(0.25) * ep * b[bt](p[0], p[1], p[2]);
                        Real const c = Real(0.25) * ep * b[bt](q[0], q[1], q[2]);
                        sum += sign * (a + c);
                        absolute += std::abs(a) + std::abs(c);
                    }
                }
                return {sum, absolute};
            };
        });
        result[0] *= (high ? factor : -factor);
        result[1] *= std::abs(factor);
        return result;
    }
}

struct NativeYeeEnergyBalance::Impl
{
    WarpX& warpx;
    std::ofstream stream;
    Copies before_b;
    Copies midpoint_e;
    Copies raw_j;
    std::map<std::string, Real> row;
    std::map<std::string, Real> cumulative;
    std::vector<std::string> columns;
    std::vector<Real> previous_k;
    bool have_previous = false;
    Real previous_u = 0;
    Real last_u = 0;
    Real start_u = 0;
    Real end_u = 0;
    Real dt = 0;
    Real volume = 0;
    amrex::GpuArray<Real, 3> inv_dx{};

    Impl (WarpX& wx, std::string const& path) : warpx(wx)
    {
        auto const dx = warpx.Geom(0).CellSizeArray();
        volume = dx[0] * dx[1] * dx[2];
        for (int d = 0; d < 3; ++d) { inv_dx[d] = Real(1) / dx[d]; }
        previous_k.resize(warpx.GetPartContainer().nSpecies(), Real(0));
        if (amrex::ParallelDescriptor::IOProcessor()) {
            auto const output = std::filesystem::path(path);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!std::filesystem::exists(output),
                "Native Yee energy CSV already exists; use a distinct segment output path");
            if (output.has_parent_path()) {
                std::filesystem::create_directories(output.parent_path());
            }
            stream.open(path);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(stream.good(), "Cannot open native Yee energy CSV");
            stream << std::setprecision(std::numeric_limits<Real>::max_digits10);
        }
    }

    Fields Get (warpx::fields::FieldType field) const
    {
        using ablastr::fields::Direction;
        auto& fields = warpx.GetMultiFabRegister();
        return {fields.get(field, Direction{0}, 0), fields.get(field, Direction{1}, 0),
                fields.get(field, Direction{2}, 0)};
    }
    Fields E () const { return Get(warpx::fields::FieldType::Efield_fp); }
    Fields B () const { return Get(warpx::fields::FieldType::Bfield_fp); }
    Fields J () const { return Get(warpx::fields::FieldType::current_fp); }

    void Record (std::string const& name, Real value, bool accumulate = false)
    {
        row[name] = value;
        if (accumulate) {
            cumulative[name] += value;
            row["cum_" + name] = cumulative[name];
        }
    }
    void Product (std::string const& name, Pair const& value)
    {
        Record(name + "_J", value[0], true);
        Record(name + "_abs_product_J", value[1], true);
    }
    Real Energy (std::string const& label)
    {
        auto const ue = Dot(E(), E(), true, volume * PhysConst::epsilon_0 / Real(2));
        auto const ub = Dot(B(), B(), false, volume / (Real(2) * PhysConst::mu0));
        Record("UE_" + label + "_J", ue[0]);
        Record("UB_" + label + "_J", ub[0]);
        Record("U_" + label + "_J", ue[0] + ub[0]);
        return ue[0] + ub[0];
    }
    void Exchange (std::string const& name, std::string const& label)
    {
        Real const next = Energy(label);
        Record(name + "_J", next - last_u, true);
        last_u = next;
    }
    void ParticleEnergy (std::string const& label)
    {
        Real total_k = 0;
        Real total_weight = 0;
        auto& particles = warpx.GetPartContainer();
        auto const names = particles.GetSpeciesNames();
        for (int s = 0; s < particles.nSpecies(); ++s) {
            auto const [energy, weight] =
                particles.GetParticleContainer(s).sumParticleWeightAndEnergy(false);
            Record("K_" + label + "_" + names[s] + "_J", energy);
            Record("weight_" + label + "_" + names[s], weight);
            total_k += energy;
            total_weight += weight;
        }
        Record("K_" + label + "_J", total_k);
        Record("weight_" + label, total_weight);
    }
};

NativeYeeEnergyBalance::NativeYeeEnergyBalance (WarpX& wx, std::string const& path)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(wx.maxLevel() == 0 && !wx.m_do_subcycling
        && wx.evolve_scheme == EvolveScheme::Explicit
        && wx.electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
        && wx.electrostatic_solver_id == ElectrostaticSolverAlgo::None
        && wx.m_em_solver_medium == MediumForEM::Vacuum
        && wx.grid_type == ablastr::utils::enums::GridType::Staggered
        && !wx.do_current_centering && !wx.do_moving_window && !EB::enabled()
        && !wx.do_dive_cleaning && !wx.do_divb_cleaning
        && !wx.do_pml_dive_cleaning && !wx.do_pml_divb_cleaning
        && wx.gamma_boost == Real(1) && !wx.DoFluidSpecies()
        && wx.m_v_galilean[0] == Real(0) && wx.m_v_galilean[1] == Real(0)
        && wx.m_v_galilean[2] == Real(0),
        "NativeYeeEnergyBalance requires single-level, nonmoving Cartesian vacuum explicit "
        "Yee, staggered grids, no EB, divergence cleaning, fluids or current centering");
    // These hooks inventory the physical domain, not particles inside an
    // extended PML. A later in-domain-PML instrument needs a separate contract.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!wx.do_pml_in_domain && !wx.pml_has_particles
        && !wx.do_pml_j_damping, "NativeYeeEnergyBalance supports exterior PML only");
    auto& particles = wx.GetPartContainer();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(particles.nLasers() == 0,
        "NativeYeeEnergyBalance does not support laser particle containers");
    for (int d = 0; d < 3; ++d) {
        for (auto const bc : {wx.particle_boundary_lo[d], wx.particle_boundary_hi[d]}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bc == ParticleBoundaryType::Periodic
                || bc == ParticleBoundaryType::Absorbing,
                "NativeYeeEnergyBalance requires periodic or absorbing particle faces");
        }
        for (auto const bc : {wx.field_boundary_lo[d], wx.field_boundary_hi[d]}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(bc == FieldBoundaryType::Periodic
                || bc == FieldBoundaryType::PML,
                "NativeYeeEnergyBalance supports periodic or exterior-PML field faces only");
        }
    }
    for (auto const& name : particles.GetSpeciesNames()) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(name.find_first_of(",\n\r") == std::string::npos,
            "NativeYeeEnergyBalance species names must be CSV-safe");
        amrex::ParmParse pp(name);
        for (auto const* key : {"reflection_model_xlo(E)", "reflection_model_xhi(E)",
             "reflection_model_ylo(E)", "reflection_model_yhi(E)",
             "reflection_model_zlo(E)", "reflection_model_zhi(E)"}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!pp.contains(key),
                "NativeYeeEnergyBalance requires default zero absorbing reflection probability");
        }
    }
    m_impl = std::make_unique<Impl>(wx, path);
}

NativeYeeEnergyBalance::~NativeYeeEnergyBalance () = default;

void NativeYeeEnergyBalance::BeginStep (int step, Real time, Real delta_t)
{
    auto& m = *m_impl;
    m.row.clear();
    m.dt = delta_t;
    m.Record("step", Real(step + 1));
    m.Record("time_start_s", time);
    m.Record("time_end_s", time + delta_t);
    m.Record("dt_s", delta_t);
    m.Record("real_epsilon", std::numeric_limits<Real>::epsilon());
    m.Record("domain_cells", Real(m.warpx.Geom(0).Domain().numPts()));
    m.ParticleEnergy("pre_push");
    Real gap = 0;
    auto const names = m.warpx.GetPartContainer().GetSpeciesNames();
    for (int s = 0; s < static_cast<int>(names.size()); ++s) {
        Real const current = m.row.at("K_pre_push_" + names[s] + "_J");
        Real const change = m.have_previous ? current - m.previous_k[s] : Real(0);
        m.Record("K_gap_" + names[s] + "_J", change, true);
        gap += change;
    }
    m.Record("K_gap_J", gap, true);
}

void NativeYeeEnergyBalance::AfterParticlePush ()
{
    auto& m = *m_impl;
    m.ParticleEnergy("post_push");
    m.Record("K_push_work_J", m.row.at("K_post_push_J") - m.row.at("K_pre_push_J"), true);
    for (auto const& name : m.warpx.GetPartContainer().GetSpeciesNames()) {
        m.Record("K_push_work_" + name + "_J", m.row.at("K_post_push_" + name + "_J")
            - m.row.at("K_pre_push_" + name + "_J"), true);
    }
}

void NativeYeeEnergyBalance::CopyRawCurrent ()
{
    auto& m = *m_impl;
    Copy(m.raw_j, m.J());
    for (int d = 0; d < 3; ++d) {
        auto ng = m.warpx.get_ng_depos_J();
        ng.min(m.raw_j[d]->nGrowVect());
        WarpXSumGuardCells(*m.raw_j[d], m.warpx.Geom(0).periodicity(), ng, 0, 1);
    }
}

void NativeYeeEnergyBalance::BeforeFirstB ()
{
    auto& m = *m_impl;
    for (int d = 0; d < 3; ++d) {
        amrex::IntVect e_type(1);
        amrex::IntVect b_type(0);
        e_type[d] = 0;
        b_type[d] = 1;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m.E()[d]->ixType().toIntVect() == e_type
            && m.B()[d]->ixType().toIntVect() == b_type
            && m.J()[d]->ixType().toIntVect() == e_type
            && m.E()[d]->nComp() == 1 && m.B()[d]->nComp() == 1
            && m.J()[d]->nComp() == 1
            && m.E()[d]->nGrowVect().min() >= 1 && m.B()[d]->nGrowVect().min() >= 1,
            "NativeYeeEnergyBalance requires native one-component Yee fields and guards");
    }
    m.start_u = m.Energy("start");
    m.last_u = m.start_u;
    m.Record("G_gap_J", m.have_previous ? m.start_u - m.previous_u : Real(0), true);
    Copy(m.before_b, m.B());
}

void NativeYeeEnergyBalance::AfterFirstB ()
{
    auto& m = *m_impl;
    Midpoint(m.before_b, m.B());
    m.Product("CB_first", CurlDot(View(m.before_b), m.E(), false, m.inv_dx,
        -m.dt * Real(0.5) * m.volume / PhysConst::mu0));
    Real const next = m.Energy("after_B_first");
    m.Record("delta_U_B_first_J", next - m.last_u, true);
    m.last_u = next;
}

void NativeYeeEnergyBalance::AfterFirstBExchange ()
{
    auto& m = *m_impl;
    m.Exchange("G_B_first_exchange", "after_B_first_exchange");
    Copy(m.midpoint_e, m.E());
}

void NativeYeeEnergyBalance::AfterE ()
{
    auto& m = *m_impl;
    Midpoint(m.midpoint_e, m.E());
    // FillBoundary only communicates periodic/interbox values of the copy.
    // No PML exchange or physical boundary operation is applied to it.
    for (auto& e : m.midpoint_e) { e->FillBoundary(m.warpx.Geom(0).periodicity()); }
    auto const ebar = View(m.midpoint_e);
    m.Product("CE", CurlDot(ebar, m.B(), true, m.inv_dx,
        m.dt * m.volume / PhysConst::mu0));
    m.Product("W", Dot(ebar, m.J(), true, m.dt * m.volume));
    m.Product("Wraw", Dot(ebar, View(m.raw_j), true, m.dt * m.volume));
    m.Product("A", CurlDot(m.B(), ebar, false, m.inv_dx,
        m.dt * m.volume / PhysConst::mu0));
    Real flux = 0;
    Real flux_abs = 0;
    std::array<std::string, 3> const names{"x", "y", "z"};
    for (int d = 0; d < 3; ++d) {
        for (int side = 0; side < 2; ++side) {
            auto const value = Face(ebar, m.B(), m.warpx.Geom(0).Domain(), d, side == 1,
                m.dt * m.volume * m.inv_dx[d] / PhysConst::mu0);
            m.Product("F_" + names[d] + (side == 0 ? "lo" : "hi"), value);
            flux += value[0];
            flux_abs += value[1];
        }
    }
    m.Product("F", {flux, flux_abs});
    Real const next = m.Energy("after_E");
    m.Record("delta_U_E_J", next - m.last_u, true);
    m.last_u = next;
    m.Record("W_processing_J", m.row.at("W_J") - m.row.at("Wraw_J"), true);
}

void NativeYeeEnergyBalance::AfterEExchange ()
{
    m_impl->Exchange("G_E_exchange", "after_E_exchange");
}

void NativeYeeEnergyBalance::BeforeSecondB ()
{
    Copy(m_impl->before_b, m_impl->B());
}

void NativeYeeEnergyBalance::AfterSecondB ()
{
    auto& m = *m_impl;
    Midpoint(m.before_b, m.B());
    m.Product("CB_second", CurlDot(View(m.before_b), m.E(), false, m.inv_dx,
        -m.dt * Real(0.5) * m.volume / PhysConst::mu0));
    Real const next = m.Energy("after_B_second");
    m.Record("delta_U_B_second_J", next - m.last_u, true);
    m.last_u = next;
}

void NativeYeeEnergyBalance::AfterFinalExchange ()
{
    auto& m = *m_impl;
    m.Exchange("G_final_exchange", "end");
    m.end_u = m.last_u;
    Real const delta_u = m.end_u - (m.have_previous ? m.previous_u : m.start_u);
    Real const g = m.row.at("G_gap_J") + m.row.at("G_B_first_exchange_J")
        + m.row.at("G_E_exchange_J") + m.row.at("G_final_exchange_J");
    Real const cb = m.row.at("CB_first_J") + m.row.at("CB_second_J");
    Real const ce = m.row.at("CE_J");
    Real const a = m.row.at("A_J");
    Real const w = m.row.at("W_J");
    Real const f = m.row.at("F_J");
    m.Record("delta_U_solve_J", m.end_u - m.start_u, true);
    m.Record("delta_U_J", delta_u, true);
    m.Record("G_J", g, true);
    m.Record("T_J", a + cb, true);
    m.Record("residual_field_update_J", delta_u + w - ce - cb - g, true);
    m.Record("residual_face_identity_J", ce - a + f, true);
    m.Record("residual_combined_J", delta_u + w + f - (a + cb) - g, true);
    m.Record("residual_B_first_J", m.row.at("delta_U_B_first_J")
        - m.row.at("CB_first_J"), true);
    m.Record("residual_E_J", m.row.at("delta_U_E_J") + w - ce, true);
    m.Record("residual_B_second_J", m.row.at("delta_U_B_second_J")
        - m.row.at("CB_second_J"), true);
    m.Record("particle_grid_mismatch_J", m.row.at("K_push_work_J") - w, true);
    Real inventory_scale = 0;
    for (auto const* stage : {"start", "after_B_first", "after_B_first_exchange",
                             "after_E", "after_E_exchange", "after_B_second", "end"}) {
        inventory_scale += m.row.at(std::string("U_") + stage + "_J");
    }
    if (m.have_previous) { inventory_scale += m.previous_u; }
    Real const scale = inventory_scale + m.row.at("W_abs_product_J")
        + m.row.at("CE_abs_product_J") + m.row.at("CB_first_abs_product_J")
        + m.row.at("CB_second_abs_product_J");
    m.Record("field_update_absolute_scale_J", scale, true);
    m.Record("face_identity_absolute_scale_J", m.row.at("CE_abs_product_J")
        + m.row.at("A_abs_product_J") + m.row.at("F_abs_product_J"), true);
}

void NativeYeeEnergyBalance::BeforeParticleBoundary ()
{
    auto& m = *m_impl;
    m.ParticleEnergy("pre_boundary");
    m.Record("K_push_to_boundary_gap_J", m.row.at("K_pre_boundary_J")
        - m.row.at("K_post_push_J"), true);
    auto& particles = m.warpx.GetPartContainer();
    auto const names = particles.GetSpeciesNames();
    auto const lo = m.warpx.Geom(0).ProbLoArray();
    auto const hi = m.warpx.Geom(0).ProbHiArray();
    amrex::GpuArray<int, 3> absorbing_lo{};
    amrex::GpuArray<int, 3> absorbing_hi{};
    for (int d = 0; d < 3; ++d) {
        absorbing_lo[d] = !m.warpx.Geom(0).isPeriodic(d)
            && WarpX::particle_boundary_lo[d] == ParticleBoundaryType::Absorbing;
        absorbing_hi[d] = !m.warpx.Geom(0).isPeriodic(d)
            && WarpX::particle_boundary_hi[d] == ParticleBoundaryType::Absorbing;
    }
    Real total_energy = 0;
    Real total_weight = 0;
    for (int s = 0; s < particles.nSpecies(); ++s) {
        auto const& pc = particles.GetParticleContainer(s);
        Real const mass = pc.getMass();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(mass > Real(0),
            "NativeYeeEnergyBalance outgoing inventory requires massive species");
        using Particle = WarpXParticleContainer::SuperParticleType;
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> ops;
        auto const result = amrex::ParticleReduce<amrex::ReduceData<Real, Real>>(
            pc, [=] AMREX_GPU_DEVICE (Particle const& p) noexcept -> Tuple
            {
                bool outgoing = false;
                for (int d = 0; d < 3; ++d) {
                    // Exact predicates in ParticleBoundaries_K.H. An edge or
                    // corner exit contributes once through the boolean union.
                    outgoing = outgoing || (absorbing_lo[d] && p.pos(d) < lo[d])
                        || (absorbing_hi[d] && p.pos(d) > hi[d]);
                }
                if (!outgoing) { return {Real(0), Real(0)}; }
                Real const weight = p.rdata(PIdx::w);
                return {weight * Algorithms::KineticEnergy(p.rdata(PIdx::ux),
                    p.rdata(PIdx::uy), p.rdata(PIdx::uz), mass), weight};
            }, ops);
        Pair outgoing{amrex::get<0>(result), amrex::get<1>(result)};
        amrex::ParallelDescriptor::ReduceRealSum(outgoing.data(), 2);
        m.Record("K_outgoing_" + names[s] + "_J", outgoing[0], true);
        m.Record("weight_outgoing_" + names[s], outgoing[1], true);
        total_energy += outgoing[0];
        total_weight += outgoing[1];
    }
    m.Record("K_outgoing_J", total_energy, true);
    m.Record("weight_outgoing", total_weight, true);
}

void NativeYeeEnergyBalance::AfterParticleBoundary ()
{
    auto& m = *m_impl;
    m.ParticleEnergy("post_boundary");
    auto const names = m.warpx.GetPartContainer().GetSpeciesNames();
    for (int s = 0; s < static_cast<int>(names.size()); ++s) {
        Real const after = m.row.at("K_post_boundary_" + names[s] + "_J");
        m.Record("residual_particle_boundary_" + names[s] + "_J",
            m.row.at("K_pre_boundary_" + names[s] + "_J") - after
            - m.row.at("K_outgoing_" + names[s] + "_J"), true);
        m.previous_k[s] = after;
    }
    m.Record("residual_particle_boundary_J", m.row.at("K_pre_boundary_J")
        - m.row.at("K_post_boundary_J") - m.row.at("K_outgoing_J"), true);
    m.Record("residual_particle_weight_boundary", m.row.at("weight_pre_boundary")
        - m.row.at("weight_post_boundary") - m.row.at("weight_outgoing"), true);
    if (m.columns.empty()) {
        for (auto const& [name, value] : m.row) {
            amrex::ignore_unused(value);
            m.columns.push_back(name);
        }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (std::size_t i = 0; i < m.columns.size(); ++i) {
                if (i != 0) { m.stream << ','; }
                m.stream << m.columns[i];
            }
            m.stream << '\n';
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m.columns.size() == m.row.size(),
        "Native Yee energy row schema changed during a segment");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (std::size_t i = 0; i < m.columns.size(); ++i) {
            if (i != 0) { m.stream << ','; }
            m.stream << m.row.at(m.columns[i]);
        }
        m.stream << '\n';
        m.stream.flush();
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m.stream.good(), "Failed writing native Yee energy CSV");
    }
    m.previous_u = m.end_u;
    m.have_previous = true;
}
#else
struct NativeYeeEnergyBalance::Impl {};
NativeYeeEnergyBalance::NativeYeeEnergyBalance (WarpX&, std::string const&)
{
    WARPX_ABORT_WITH_MESSAGE("NativeYeeEnergyBalance requires a Cartesian 3D build");
}
NativeYeeEnergyBalance::~NativeYeeEnergyBalance () = default;
void NativeYeeEnergyBalance::BeginStep (int, amrex::Real, amrex::Real) {}
void NativeYeeEnergyBalance::AfterParticlePush () {}
void NativeYeeEnergyBalance::CopyRawCurrent () {}
void NativeYeeEnergyBalance::BeforeFirstB () {}
void NativeYeeEnergyBalance::AfterFirstB () {}
void NativeYeeEnergyBalance::AfterFirstBExchange () {}
void NativeYeeEnergyBalance::AfterE () {}
void NativeYeeEnergyBalance::AfterEExchange () {}
void NativeYeeEnergyBalance::BeforeSecondB () {}
void NativeYeeEnergyBalance::AfterSecondB () {}
void NativeYeeEnergyBalance::AfterFinalExchange () {}
void NativeYeeEnergyBalance::BeforeParticleBoundary () {}
void NativeYeeEnergyBalance::AfterParticleBoundary () {}
#endif
