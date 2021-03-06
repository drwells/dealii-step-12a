#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
// TODO for some reason I get incomplete types due to the
// dof_handler.active_cell_iterators() call if this header is not included
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

#include "../dg_tools/dg_tools.h"

using namespace dealii;

namespace Step12a
{
  using namespace dealii;

  template<int dim>
  inline double inflow_function(const Point<dim> &point)
  {
    return (0.25 < point[1] && point[1] < 0.75) ? 1.0 : 0.0;
    // return (point[0] < 0.5) ? 1.0 : 0.0;
  }

  template<int dim>
  inline Tensor<1, dim> convection_function(const Point<dim> &point)
  {
    // TODO there should be a way to brace initialize convection in this way
    // directly.
    Tensor<1, dim> result;
    result[0] = -point[1];
    result[1] = point[0];
    result /= result.norm();
    // result[0] = 1;
    return result;
  }

  constexpr int n_cycles {20};

  template<int dim>
  class AdvectionProblem
  {
  public:
    AdvectionProblem();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void calculate_cell_matrix
    (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &cell,
     FEValues<dim> &cell_values,
     FullMatrix<double> &cell_matrix);
    void calculate_flux_terms
    (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &current_cell,
     FEFaceValues<dim> &current_face_values,
     const TriaIterator<DoFCellAccessor<DoFHandler<dim>, false>> &neighbor_cell,
     FEFaceValuesBase<dim> &neighbor_face_values,
     FullMatrix<double> &current_to_current_flux,
     FullMatrix<double> &current_to_neighbor_flux,
     FullMatrix<double> &neighbor_to_current_flux,
     FullMatrix<double> &neighbor_to_neighbor_flux);
    void calculate_boundary_terms
    (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &current_cell,
     FEFaceValues<dim>  &current_face_values,
     FullMatrix<double> &cell_matrix,
     Vector<double>     &current_cell_rhs);
    void solve_system();
    void refine_grid();

    Triangulation<dim> triangulation;
    FE_DGQ<dim>        fe;
    DoFHandler<dim>    dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    unsigned int cycle;
  };



  template<int dim>
  AdvectionProblem<dim>::AdvectionProblem()
    : fe(1),
      dof_handler(triangulation),
      cycle {0}
  {
    GridGenerator::hyper_cube(triangulation, 0, 1);
    constexpr unsigned int n_global_refines {7};
    triangulation.refine_global(n_global_refines);
  }

  template<int dim>
  void AdvectionProblem<dim>::setup_system()
  {
    std::ofstream out("grid-" + Utilities::int_to_string(cycle) + ".eps");
    GridOut grid_out;
    grid_out.write_eps(triangulation, out);

    dof_handler.distribute_dofs(fe);

    {
      DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs());
      DoFTools::make_flux_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
      sparsity_pattern.copy_from(dynamic_sparsity_pattern);
    }

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  template<int dim>
  void AdvectionProblem<dim>::calculate_cell_matrix
  (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &cell,
   FEValues<dim> &cell_values,
   FullMatrix<double> &cell_matrix)
  {
    for (unsigned int q_point_n = 0; q_point_n < cell_values.n_quadrature_points;
         ++q_point_n)
      {
        const auto convection = convection_function
          (cell_values.quadrature_point(q_point_n));

        for (unsigned int i = 0; i < cell_values.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < cell_values.dofs_per_cell; ++j)
              {
                cell_matrix(i, j) -= convection
                  * cell_values.shape_grad(i, q_point_n)
                  * cell_values.shape_value(j, q_point_n)
                  * cell_values.JxW(q_point_n);
              }
          }
      }

    std::vector<types::global_dof_index> local_dof_indices
      (cell_values.dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
    system_matrix.add(local_dof_indices, cell_matrix);
  }



  /*
   * Note that neighbor_face_values may be of type FEFaceValues or
   * FESubfaceValues, depending on whether or not the current cell is at a
   * higher level than the neighbor cell.
   */
  template<int dim>
  void AdvectionProblem<dim>::calculate_flux_terms
  (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &current_cell,
   FEFaceValues<dim> &current_face_values,
   const TriaIterator<DoFCellAccessor<DoFHandler<dim>, false>> &neighbor_cell,
   FEFaceValuesBase<dim> &neighbor_face_values,
   FullMatrix<double> &current_to_current_flux,
   FullMatrix<double> &current_to_neighbor_flux,
   FullMatrix<double> &neighbor_to_current_flux,
   FullMatrix<double> &neighbor_to_neighbor_flux)
  {
    Assert(current_face_values.n_quadrature_points == neighbor_face_values.n_quadrature_points,
           ExcMessage("The two quadrature rules should have the same number of points."));

    const auto normal = current_face_values.get_all_normal_vectors()[0];
    for (unsigned int q_point_n = 0; q_point_n < current_face_values.n_quadrature_points;
         ++q_point_n)
      {
        Assert((current_face_values.quadrature_point(q_point_n)
                - neighbor_face_values.quadrature_point(q_point_n)).norm() < 1e-15,
               ExcMessage("The quadrature points should be in the same positions."));

        const auto convection = convection_function
          (current_face_values.quadrature_point(q_point_n));

        /*
         * Note that if we are integrating the neighbor's test functions, then
         * we are really performing the integration on the neighbor's face,
         * which has a normal vector pointing in the opposite direction as the
         * current cell's normal vector.
         */
        if (normal * convection > 0) // the current cell is upwind at the point
          {
            for (unsigned int i = 0; i < current_face_values.dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < neighbor_face_values.dofs_per_cell; ++j)
                  {
                    current_to_current_flux(i, j) += convection * normal
                      * current_face_values.shape_value(i, q_point_n)
                      * current_face_values.shape_value(j, q_point_n)
                      * current_face_values.JxW(q_point_n);

                    current_to_neighbor_flux(i, j) -= convection * normal
                      // Note that rows correspond to test functions, so we
                      // use the neighbor's test functions here (values of the
                      // solution come from the current cell)
                      * neighbor_face_values.shape_value(i, q_point_n)
                      * current_face_values.shape_value(j, q_point_n)
                      * current_face_values.JxW(q_point_n);
                  }
              }
          }
        // Otherwise, use the neighbor values as the upwind values.
        else
          {
            for (unsigned int i = 0; i < current_face_values.dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < neighbor_face_values.dofs_per_cell; ++j)
                  {
                    neighbor_to_current_flux(i, j) += convection * normal
                      // See the comment above regarding is and js: values of
                      // the solution now lie on the neighboring cell.
                      * current_face_values.shape_value(i, q_point_n)
                      * neighbor_face_values.shape_value(j, q_point_n)
                      * current_face_values.JxW(q_point_n);

                    neighbor_to_neighbor_flux(i, j) -= convection * normal
                      * neighbor_face_values.shape_value(i, q_point_n)
                      * neighbor_face_values.shape_value(j, q_point_n)
                      * current_face_values.JxW(q_point_n);
                  }
              }
          }
      }

    std::vector<types::global_dof_index> current_dofs
      (current_face_values.dofs_per_cell);
    current_cell->get_dof_indices(current_dofs);
    std::vector<types::global_dof_index> neighbor_dofs
      (neighbor_face_values.dofs_per_cell);
    neighbor_cell->get_dof_indices(neighbor_dofs);

    // TODO double check this system matrix addition. The rows of the system
    // matrix correspond to test functions.
    system_matrix.add(current_dofs, current_to_current_flux);
    system_matrix.add(neighbor_dofs, current_dofs, current_to_neighbor_flux);

    system_matrix.add(neighbor_dofs, neighbor_to_neighbor_flux);
    system_matrix.add(current_dofs, neighbor_dofs, neighbor_to_current_flux);
  }



  template<int dim>
  void AdvectionProblem<dim>::calculate_boundary_terms
  (const TriaActiveIterator<DoFCellAccessor<DoFHandler<dim>, false>> &current_cell,
   FEFaceValues<dim>  &current_face_values,
   FullMatrix<double> &current_cell_matrix,
   Vector<double>     &current_cell_rhs)
  {
    const auto dofs_per_cell = fe.dofs_per_cell;
    const auto normal = current_face_values.get_all_normal_vectors()[0];

    for (unsigned int q_point_n = 0;
         q_point_n < current_face_values.n_quadrature_points;
         ++q_point_n)
      {
        const auto convection = convection_function
          (current_face_values.quadrature_point(q_point_n));

        if (convection * normal > 0) // current cell is upwind, so this is the
                                     // outflow
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    current_cell_matrix(i, j) += convection * normal
                      * current_face_values.shape_value(i, q_point_n)
                      * current_face_values.shape_value(j, q_point_n)
                      * current_face_values.JxW(q_point_n);
                  }
              }
          }
        else // inflow
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                current_cell_rhs(i) -= convection * normal
                  * inflow_function(current_face_values.quadrature_point(q_point_n))
                  * current_face_values.shape_value(i, q_point_n)
                  * current_face_values.JxW(q_point_n);
              }
          }
      }

    std::vector<types::global_dof_index> current_dofs
      (current_face_values.dofs_per_cell);
    current_cell->get_dof_indices(current_dofs);
    system_matrix.add(current_dofs, current_cell_matrix);
    system_rhs.add(current_dofs, current_cell_rhs);
  }



  template<int dim>
  void AdvectionProblem<dim>::assemble_system()
  {
    const auto dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> current_cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> current_to_current_flux(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> current_to_neighbor_flux(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> neighbor_to_current_flux(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> neighbor_to_neighbor_flux(dofs_per_cell, dofs_per_cell);
    Vector<double>     current_cell_rhs(dofs_per_cell);

    const QGauss<dim> cell_quadrature(3);
    const QGauss<dim - 1> face_quadrature(3);

    const UpdateFlags update_flags
    {update_values | update_quadrature_points | update_JxW_values};

    FEValues<dim> current_cell_values(fe, cell_quadrature,
                                      update_flags | update_gradients);
    FEFaceValues<dim> current_face_values(fe, face_quadrature, update_flags
                                          | update_normal_vectors);

    FEFaceValues<dim> neighbor_face_values(fe, face_quadrature, update_flags);
    FESubfaceValues<dim> neighbor_subface_values(fe, face_quadrature, update_flags);

    for (const auto &current_cell : dof_handler.active_cell_iterators())
      {
        // cell has type TriaActiveIterator<DoFCellAccessor<DoFHandler<2>, false> >
        if (current_cell->is_locally_owned())
          {
            current_cell_values.reinit(current_cell);
            current_cell_matrix = 0.0;
            calculate_cell_matrix(current_cell, current_cell_values, current_cell_matrix);

            for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell;
                 ++face_n)
              {
                const int neighbor_index {current_cell->neighbor_index(face_n)};
                if (neighbor_index != -1) // interior face
                  {
                    // for DG we need to access the FE space on the adjacent cell.
                    auto neighbor_cell {current_cell->neighbor(face_n)};

                    bool do_face_integration {false};
                    bool neighbor_is_level_lower {false};

                    /*
                     * Always integrate if the current cell is more refined
                     * than the neighbor.
                     */
                    if (current_cell->level() > neighbor_cell->level())
                      {
                        do_face_integration = true;
                        neighbor_is_level_lower = true;
                      }

                    /*
                     * If neighbor_cell is not active, then the neighboring
                     * cell is at a higher refinement level (and therefore we
                     * should not do the face integral right now).
                     */
                    else if (neighbor_cell->active() && neighbor_cell < current_cell)
                      {
                        do_face_integration = true;
                      }

                    if (do_face_integration)
                      {
                        /*
                         * Unfortunately, FEFaceValuesBase does not permit the
                         * creation of references (I may not fully understand
                         * why), so use a pointer.
                         */
                        FEFaceValuesBase<dim> *neighbor_face_rule = &neighbor_face_values;

                        /*
                         * If the neighbor is on a higher level then we need
                         * to use a special subface quadrature rule.
                         */
                        const unsigned int neighbor_face_n
                        {DGTools::calculate_neighbor_face_n
                            (current_cell, neighbor_cell, face_n)};
                        if (neighbor_is_level_lower)
                          {
                            const unsigned int neighbor_subface_n
                            {DGTools::calculate_subface_n
                                (current_cell, neighbor_cell, face_n)};

                            neighbor_face_rule = &neighbor_subface_values;
                            neighbor_subface_values.reinit
                              (neighbor_cell, neighbor_face_n, neighbor_subface_n);
                          }
                        else
                          {
                            neighbor_face_values.reinit(neighbor_cell, neighbor_face_n);
                          }

                        current_face_values.reinit(current_cell, face_n);
                        current_to_current_flux = 0.0;
                        current_to_neighbor_flux = 0.0;
                        neighbor_to_current_flux = 0.0;
                        neighbor_to_neighbor_flux = 0.0;
                        calculate_flux_terms
                          (current_cell, current_face_values, neighbor_cell, *neighbor_face_rule,
                           current_to_current_flux, current_to_neighbor_flux,
                           neighbor_to_current_flux, neighbor_to_neighbor_flux);
                      }
                  }
                else // boundary face: always integrate
                  {
                    current_cell_matrix = 0.0;
                    current_cell_rhs = 0.0;
                    current_face_values.reinit(current_cell, face_n);
                    calculate_boundary_terms(current_cell, current_face_values,
                                             current_cell_matrix, current_cell_rhs);
                  }
              }
          }
      }
  }



  template<int dim>
  void AdvectionProblem<dim>::solve_system()
  {
    SparseDirectUMFPACK direct;
    direct.initialize(system_matrix);
    direct.vmult(solution, system_rhs);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    data_out.build_patches();

    std::ofstream vtu_output("result-" + Utilities::int_to_string(cycle) + ".vtu");
    data_out.write_vtu(vtu_output);
  }



  template<int dim>
  void AdvectionProblem<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 2),
                                       typename FunctionMap<dim>::type(),
                                       solution,
                                       estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_fraction
      (triangulation, estimated_error_per_cell, 0.3, 0.1);

    triangulation.execute_coarsening_and_refinement();
  }



  template<int dim>
  void AdvectionProblem<dim>::run()
  {
    for (; cycle < n_cycles; ++cycle)
      {
        setup_system();
        assemble_system();
        solve_system();
        refine_grid();
      }
  }
}

int main()
{
  constexpr int dim {2};
  Step12a::AdvectionProblem<dim> advection_problem;
  advection_problem.run();
}
