#include "dg_tools.h"

constexpr int dim {2};

int main()
{
  using namespace dealii;

  Triangulation<dim> triangulation;
  FE_DGQ<dim> fe(1);

  DoFHandler<dim> dof_handler(triangulation);
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(1);

  // refine the bottom left
  {
    triangulation.begin_active()->set_refine_flag();
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }

  dof_handler.distribute_dofs(fe);

  const QGauss<dim - 1> face_quadrature(2);

  const UpdateFlags update_flags
    {update_values | update_quadrature_points | update_JxW_values};

  FEFaceValues<dim> current_face_values(fe, face_quadrature, update_flags
                                        | update_normal_vectors);
  FEFaceValues<dim> neighbor_face_values(fe, face_quadrature, update_flags);
  FESubfaceValues<dim> neighbor_subface_values(fe, face_quadrature, update_flags);

  for (const auto &current_cell : dof_handler.active_cell_iterators())
    {
      // cell has type TriaActiveIterator<DoFCellAccessor<DoFHandler<2>, false> >
      if (current_cell->is_locally_owned())
        {
          for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell;
               ++face_n)
            {
              const int neighbor_index {current_cell->neighbor_index(face_n)};
              if (neighbor_index != -1) // interior face
                {
                  auto neighbor_cell {current_cell->neighbor(face_n)};
                  std::cout << "(" << current_cell->level() << ", "
                            << current_cell->index() << ")"
                            << " > "
                            << "(" << neighbor_cell->level() << ", "
                            << neighbor_cell->index() << ")"
                            << std::endl;

                  const bool neighbor_is_level_lower
                    {current_cell->level() > neighbor_cell->level()};

                  FEValuesBase<dim> *neighbor_face_rule = &neighbor_face_values;

                  const unsigned int neighbor_face_n
                  {DGTools::calculate_neighbor_face_n(current_cell, neighbor_cell,
                                                      face_n)};
                  std::cout << "neighbor_face_n = " << neighbor_face_n << std::endl;
                  if (neighbor_is_level_lower)
                    {
                      const unsigned int neighbor_subface_n
                      {DGTools::calculate_subface_n(current_cell, neighbor_cell,
                                                    face_n)};
                      std::cout << "neighbor_subface_n = "
                                << neighbor_subface_n
                                << std::endl;

                      neighbor_subface_values.reinit
                        (neighbor_cell, neighbor_face_n, neighbor_subface_n);
                      neighbor_face_rule = &neighbor_subface_values;
                    }
                  else
                    {
                      neighbor_face_values.reinit(neighbor_cell, neighbor_face_n);
                    }

                  current_face_values.reinit(current_cell, face_n);
                }
            }
        }
    }
}
