/*!
 * \file solver_config.h
 * \brief Configuration keys for optional layout candidate composition.
 */
#ifndef TVM_TL_LAYOUT_SOLVER_CONFIG_H_
#define TVM_TL_LAYOUT_SOLVER_CONFIG_H_

namespace tvm {
namespace tl {
static constexpr const char *kLayoutSolver = "tl.layout_solver";
static constexpr const char *kLayoutSolverTimeoutMs =
    "tl.layout_solver_timeout_ms";
static constexpr const char *kLayoutSolverVerbose = "tl.layout_solver_verbose";
} // namespace tl
} // namespace tvm

#endif // TVM_TL_LAYOUT_SOLVER_CONFIG_H_
