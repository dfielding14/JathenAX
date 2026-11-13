// AthenaK problem generator for a one-timestep, double-precision comparison.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"

namespace {

void WriteStage(Mesh *mesh, const Real stage_dt) {
  static_assert(sizeof(Real) == sizeof(double), "Use double-precision AthenaK");
  auto *pack = mesh->pmb_pack;
  auto *hydro = pack->phydro;
  const auto &cells = mesh->mb_indcs;
  const auto &size = pack->pmb->mb_size.h_view(0);
  const auto &eos = hydro->peos->eos_data;
  const auto reconstruct = hydro->recon_method;

  // This callback runs after the RK update and before primitive recovery.
  const auto w = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hydro->w0);
  const auto fx = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hydro->uflx.x1f);
  const auto fy = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hydro->uflx.x2f);
  const auto saved = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hydro->u1);
  const auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), hydro->u0);

  static int stage = 0;
  std::ofstream out("stage" + std::to_string(++stage) + ".txt");
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(std::numeric_limits<Real>::max_digits10)
      << cells.nx2 << ' ' << cells.nx1 << ' ' << size.dx1 << ' ' << size.dx2 << ' '
      << eos.iso_cs << ' ' << eos.dfloor << ' ' << mesh->dt << ' ' << stage_dt << '\n';

  const int k = cells.ks;
  for (int j = cells.js; j <= cells.je; ++j) {
    for (int i = cells.is; i <= cells.ie; ++i) {
      const auto write_cell = [&](const auto &array) {
        for (int n = IDN; n <= IM2; ++n) {
          out << array(0,n,k,j,i) << ' ';
        }
      };
      write_cell(w);
      // Reuse the scalar reconstruction routines behind the actual fluxes.
      for (int axis = 0; axis < 2; ++axis) {
        Real left[3], right[3], unused;
        for (int n = IDN; n <= IM2; ++n) {
          const auto q = [&](int offset) {
            return w(0,n,k,j + (axis == 1)*offset,i + (axis == 0)*offset);
          };
          switch (reconstruct) {
            case ReconstructionMethod::dc:
              left[n] = q(-1);
              right[n] = q(0);
              break;
            case ReconstructionMethod::plm:
              PLM(q(-2), q(-1), q(0), left[n], unused);
              PLM(q(-1), q(0), q(1), unused, right[n]);
              break;
            case ReconstructionMethod::ppm4:
              PPM4(q(-3), q(-2), q(-1), q(0), q(1), left[n], unused);
              PPM4(q(-2), q(-1), q(0), q(1), q(2), unused, right[n]);
              break;
            case ReconstructionMethod::ppmx:
              PPMX(q(-3), q(-2), q(-1), q(0), q(1), left[n], unused);
              PPMX(q(-2), q(-1), q(0), q(1), q(2), unused, right[n]);
              break;
            case ReconstructionMethod::wenoz:
              WENOZ(q(-3), q(-2), q(-1), q(0), q(1), left[n], unused);
              WENOZ(q(-2), q(-1), q(0), q(1), q(2), unused, right[n]);
              break;
          }
          if (n == IDN && (reconstruct == ReconstructionMethod::ppmx
                          || reconstruct == ReconstructionMethod::wenoz)) {
            left[n] = fmax(left[n], eos.dfloor);
            right[n] = fmax(right[n], eos.dfloor);
          }
        }
        for (Real value : left) out << value << ' ';
        for (Real value : right) out << value << ' ';
      }
      write_cell(fx);
      write_cell(fy);
      write_cell(saved);
      write_cell(u);
      out << '\n';
    }
  }
}

}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart || pmy_mesh_->nmb_total != 1 || !pmy_mesh_->two_d) {
    throw std::runtime_error("The stage comparison requires a fresh 2D single-block run");
  }
  user_srcs_func = WriteStage;
  auto &u = pmy_mesh_->pmb_pack->phydro->u0;
  const auto cells = pmy_mesh_->mb_indcs;
  const bool jump = pin->GetBoolean("problem", "velocity_jump");
  const Real twopi = 2.0 * M_PI;
  par_for("reference_initial_state", DevExeSpace(), 0, 0, cells.ks, cells.ke,
          cells.js, cells.je, cells.is, cells.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = twopi * (i - cells.is + 0.5) / cells.nx1;
    const Real y = twopi * (j - cells.js + 0.5) / cells.nx2;
    const Real rho = 1.0 + 0.2*sin(x) + 0.1*cos(x + y);
    // The jump exercises supersonic upwinding and Roe's LLF fallback.
    const Real vx = jump ? ((i - cells.is < cells.nx1/2) ? -2.5 : 2.5)
                         : 0.2*cos(y) + 0.1*sin(x - y);
    const Real vy = 0.15*sin(x) + 0.2*cos(x + y);
    u(m,IDN,k,j,i) = rho;
    u(m,IM1,k,j,i) = rho*vx;
    u(m,IM2,k,j,i) = rho*vy;
    u(m,IM3,k,j,i) = 0.0;
  });
}
