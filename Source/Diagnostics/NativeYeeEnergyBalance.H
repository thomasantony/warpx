/* Copyright 2026 WarpX contributors
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NATIVE_YEE_ENERGY_BALANCE_H_
#define WARPX_NATIVE_YEE_ENERGY_BALANCE_H_

#include <AMReX_REAL.H>

#include <memory>
#include <string>

class WarpX;

/** Passive, single-level Cartesian Yee stage accounting.
 * This is deliberately not a ReducedDiag: requesting its output must never
 * trigger additional particle momentum synchronization pushes.
 */
class NativeYeeEnergyBalance
{
public:
    NativeYeeEnergyBalance (WarpX& warpx, std::string const& path);
    ~NativeYeeEnergyBalance ();
    void BeginStep (int step, amrex::Real time, amrex::Real dt);
    void AfterParticlePush ();
    void CopyRawCurrent ();
    void BeforeFirstB ();
    void AfterFirstB ();
    void AfterFirstBExchange ();
    void AfterE ();
    void AfterEExchange ();
    void BeforeSecondB ();
    void AfterSecondB ();
    void AfterFinalExchange ();
    void BeforeParticleBoundary ();
    void AfterParticleBoundary ();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif
