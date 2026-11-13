// Dump the single-block forcing comparison at full precision.
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {

std::ofstream TraceText(const std::string &name) {
  std::ofstream out(name, std::ios::app);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out << std::setprecision(17);
  return out;
}

void TraceField(const Mesh *mesh, const DvceArray5D<Real> &field,
                const std::string &name, int components) {
  static_assert(sizeof(Real) == sizeof(double), "Use double-precision AthenaK");
  if (mesh->nmb_total != 1 || !mesh->two_d) {
    throw std::runtime_error("Use a single 2D MeshBlock for the forcing trace");
  }
  const auto &cells = mesh->mb_indcs;
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), field);
  std::ofstream out(name + "_" + std::to_string(mesh->ncycle) + ".bin",
                    std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  for (int j = cells.js; j <= cells.je; ++j) {
    for (int i = cells.is; i <= cells.ie; ++i) {
      for (int n = 0; n < components; ++n) {
        const Real value = host(0,n,cells.ks,j,i);
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
      }
    }
  }
}

}  // namespace
