// Float64 reference window for a single periodic, driven and cooled AthenaK block.
#include <cmath>
#include <filesystem>
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
#include "srcterms/turb_driver.hpp"

namespace {
constexpr int ncons = 5;
// The two cell states contain (rho, mx, my, mz, E, thermal energy, SGS energy).
enum Field {initial = 0, final_state = 7, face_x = 14, face_y = 19, face_z = 24,
            impulse = 29, work = 32, cooling = 33, nfields = 34};
constexpr Real rk_weight[] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
Real tau_cool;
int floors_at_start;

int FloorCount(Mesh *mesh) {
  const auto &e = mesh->ecounter;
  return e.neos_dfloor + e.neos_efloor + e.neos_tfloor;
}

void Cool(Mesh *mesh, Real beta_dt) {
  auto *hydro = mesh->pmb_pack->phydro;
  const auto u = hydro->u0, w = hydro->w0;
  const auto c = mesh->mb_indcs;
  const Real fraction = beta_dt/tau_cool;
  par_for("linear_cooling", DevExeSpace(), c.ks,c.ke, c.js,c.je, c.is,c.ie,
  KOKKOS_LAMBDA(int k, int j, int i) {
    u(0,IEN,k,j,i) -= fraction*w(0,IEN,k,j,i);
  });
}

// Restrict the endpoints independently of the fluxes. At the final endpoint,
// the measured acceleration and post-kick momentum give the finite kick work.
void RecordState(Mesh *mesh, int r, const DvceArray4D<Real> &data, Field offset) {
  auto *pack = mesh->pmb_pack;
  const auto u = pack->phydro->u0, w = pack->phydro->w0, force = pack->pturb->force;
  const auto c = mesh->mb_indcs;
  const Real dt = mesh->dt;
  const Real cells = static_cast<Real>(r)*r*r;
  const auto size = pack->pmb->mb_size.h_view(0);
  const Real volume = size.dx1*size.dx2*size.dx3;
  par_for("window_state", DevExeSpace(), 0,c.nx3/r-1, 0,c.nx2/r-1, 0,c.nx1/r-1,
  KOKKOS_LAMBDA(int K, int J, int I) {
    Real state[7]{}, dp[3]{}, dE = 0;
    for (int z=0; z<r; ++z) for (int y=0; y<r; ++y) for (int x=0; x<r; ++x) {
      const int k=c.ks+r*K+z, j=c.js+r*J+y, i=c.is+r*I+x;
      const Real rho = u(0,IDN,k,j,i);
      for (int n=0; n<ncons; ++n) state[n] += u(0,n,k,j,i);
      state[5] += w(0,IEN,k,j,i);
      for (int d=0; d<3; ++d) {
        const Real m = u(0,IM1+d,k,j,i);
        state[6] += 0.5*m*m/rho;
        if (offset == final_state) {
          const Real dv = dt*force(0,d,k,j,i);
          dp[d] += rho*dv;
          dE += (m - 0.5*rho*dv)*dv;
        }
      }
    }
    for (int d=0; d<3; ++d) state[6] -= 0.5*state[IM1+d]*state[IM1+d]/state[IDN];
    for (int n=0; n<7; ++n) data(offset+n,K,J,I) = state[n]/cells;
    if (offset == final_state) {
      for (int d=0; d<3; ++d) data(impulse+d,K,J,I) = volume*dp[d];
      data(work,K,J,I) = volume*dE;
    }
  });
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  static_assert(sizeof(Real) == sizeof(double), "Build this fixture in float64");
  Mesh *mesh = pmy_mesh_;
  auto *pack = mesh->pmb_pack;
  const auto c = mesh->mb_indcs;
  const int r = pin->GetInteger("problem","coarsen_factor");
  const int start_cycle = pin->GetInteger("problem","window_start_cycle");
  tau_cool = pin->GetReal("problem","tau_cool");
  if (!user_srcs || !pack->phydro || !pack->phydro->peos->eos_data.is_ideal ||
      !pack->pturb || mesh->nmb_total != 1 || !mesh->three_d ||
      pin->GetString("time","integrator") != "rk3" || r<1 ||
      c.nx1%r || c.nx2%r || c.nx3%r || start_cycle < -1 ||
      !std::isfinite(tau_cool) || tau_cool<=0) {
    throw std::runtime_error("Window requires 3D single-block ideal hydro, RK3, driving, "
                             "positive cooling time and an aligned coarsening factor");
  }
  for (auto bc : mesh->mesh_bcs) {
    if (bc != BoundaryFlag::periodic) throw std::runtime_error("Use periodic boundaries");
  }
  const auto forcing = pack->pturb->RestartMetadata();
  if (forcing.turb_flag != 2 || forcing.tdriv_start > 0 ||
      forcing.normalization != static_cast<int>(TurbNormalization::edot)) {
    throw std::runtime_error("Use continuous edot forcing active from cycle zero");
  }
  user_srcs_func = Cool;
  const Real gamma = pack->phydro->peos->eos_data.gamma;
  if (!restart) {
    const auto u = pack->phydro->u0;
    // Unit density, unit sound speed and zero momentum as in AthenaK's turb pgen.
    par_for("window_ic", DevExeSpace(), c.ks,c.ke, c.js,c.je, c.is,c.ie,
    KOKKOS_LAMBDA(int k, int j, int i) {
      u(0,IDN,k,j,i) = 1.0;
      for (int n=IM1; n<=IM3; ++n) u(0,n,k,j,i) = 0.0;
      u(0,IEN,k,j,i) = 1.0/(gamma*(gamma-1.0));
    });
  }

  if (start_cycle == -1) return;  // The stationary calibration needs only box histories.
  const int nx=c.nx1/r, ny=c.nx2/r, nz=c.nx3/r;
  const auto size = pack->pmb->mb_size.h_view(0);
  const Real dx=size.dx1, dy=size.dx2, dz=size.dx3;
  const DvceArray4D<Real> data("window",nfields,nz,ny,nx);
  TaskID none(0);
  pack->tl_map["before_stagen"]->AddTask([=](Driver*, int stage) {
    if (mesh->ncycle < start_cycle) return TaskStatus::complete;
    if (stage == 1) {
      floors_at_start = FloorCount(mesh);
      Kokkos::deep_copy(data, 0.0);
      RecordState(mesh,r,data,initial);
    }
    // Effective weights in the completed SSPRK3 step differ from its stage betas.
    const Real loss_weight = rk_weight[stage-1]*mesh->dt*dx*dy*dz/tau_cool;
    const auto w = pack->phydro->w0;
    par_for("window_cooling", DevExeSpace(), 0,nz-1, 0,ny-1, 0,nx-1,
    KOKKOS_LAMBDA(int K, int J, int I) {
      Real thermal = 0;
      for (int z=0; z<r; ++z) for (int y=0; y<r; ++y) for (int x=0; x<r; ++x) {
        thermal += w(0,IEN,c.ks+r*K+z,c.js+r*J+y,c.is+r*I+x);
      }
      data(cooling,K,J,I) -= loss_weight*thermal;
    });
    return TaskStatus::complete;
  }, none);
  pack->tl_map["after_stagen"]->AddTask([=](Driver*, int stage) {
    if (mesh->ncycle < start_cycle) return TaskStatus::complete;
    const Real weight = rk_weight[stage-1]*mesh->dt;
    const auto &flux = pack->phydro->uflx;
    const auto fx=flux.x1f, fy=flux.x2f, fz=flux.x3f;
    par_for("window_face_integrals", DevExeSpace(), 0,ncons-1, 0,nz-1, 0,ny-1, 0,nx-1,
    KOKKOS_LAMBDA(int n, int K, int J, int I) {
      const int k=c.ks+r*K, j=c.js+r*J, i=c.is+r*I;
      Real sx=0, sy=0, sz=0;
      for (int a=0; a<r; ++a) for (int b=0; b<r; ++b) {
        sx += fx(0,n,k+a,j+b,i);
        sy += fy(0,n,k+a,j,i+b);
        sz += fz(0,n,k,j+a,i+b);
      }
      data(face_x+n,K,J,I) += weight*dy*dz*sx;
      data(face_y+n,K,J,I) += weight*dx*dz*sy;
      data(face_z+n,K,J,I) += weight*dx*dy*sz;
    });
    if (stage == 3) {
      if (FloorCount(mesh) != floors_at_start) {
        throw std::runtime_error("EOS floor activated during the recorded step");
      }
      RecordState(mesh,r,data,final_state);
      const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),data);
      const auto path = "window."+std::to_string(mesh->ncycle)+".bin";
      if (std::filesystem::exists(path)) throw std::runtime_error("Already exists: "+path);
      std::ofstream out(path+".tmp", std::ios::binary);
      out.exceptions(std::ios::failbit | std::ios::badbit);
      out << std::setprecision(std::numeric_limits<Real>::max_digits10)
          << nx << ' ' << ny << ' ' << nz << ' ' << r
          << ' ' << r*dx << ' ' << r*dy << ' ' << r*dz
          << ' ' << mesh->time << ' ' << mesh->dt << ' ' << forcing.dedt
          << ' ' << tau_cool << ' ' << gamma << '\n';
      out.write(reinterpret_cast<const char*>(host.data()), host.size()*sizeof(Real));
      out.close();
      std::filesystem::rename(path+".tmp", path);
    }
    return TaskStatus::complete;
  }, none);
}
