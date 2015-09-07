#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
// TODO for some reason I get incomplete types due to the
// dof_handler.active_cell_iterators() call if this header is not included
#include <deal.II/fe/mapping_q1.h>

#include <fstream>
#include <iostream>

namespace DGTools
{
  using namespace dealii;

  template<typename CurrentIterator, typename NeighborIterator>
  unsigned int calculate_neighbor_face_n
  (const CurrentIterator &current_cell,
   const NeighborIterator &neighbor_cell,
   const unsigned int current_cell_face_n)
  {
    Assert(neighbor_cell->level() <= current_cell->level(),
           ExcMessage("Under this scheme the current cell should be refined "
                      "more (or the same amount) as the neighbor cell."));
    constexpr int dim {CurrentIterator::AccessorType::dimension};

    // if the current cell is refined then take the parent's face.
    const auto current_face_iterator
      {current_cell->level() > neighbor_cell->level()
        ? current_cell->parent()->face(current_cell_face_n)
          : current_cell->face(current_cell_face_n)};

    for (unsigned int neighbor_face_n = 0; neighbor_face_n < GeometryInfo<dim>::faces_per_cell;
         ++neighbor_face_n)
      {
        if (neighbor_cell->face(neighbor_face_n) == current_face_iterator)
          {
            return neighbor_face_n;
          }
      }

    Assert(false, ExcMessage("neighbor_cell is not current_cell's neighbor."));
    return numbers::invalid_unsigned_int;
  }

  // Conventionally, these iterators should have types
  //
  // current cell: TriaActiveIterator<dealii::DoFCellAccessor<dealii::DoFHandler<2>, false>>
  // neighbor cell: TriaIterator<dealii::DoFCellAccessor<dealii::DoFHandler<2>, false>>
  //
  // it is worth investigating how to enforce this.
  template<typename CurrentIterator, typename NeighborIterator>
  unsigned int calculate_subface_n
  (const CurrentIterator &current_cell,
   const NeighborIterator &neighbor_cell,
   const int current_cell_face_n)
  {
    constexpr int dim {CurrentIterator::AccessorType::dimension};

    static_assert(dim == 2, "Only implemented in 2D");
    Assert(current_cell->active(), TriaAccessorExceptions::ExcCellNotActive());
    Assert(neighbor_cell->active(), TriaAccessorExceptions::ExcCellNotActive());
    // TODO switch this to a named exception class
    Assert(current_cell->level() > neighbor_cell->level(),
           ExcMessage("Under this scheme the current cell should be refined more "
                      "than the neighbor cell."));

    const unsigned int neighbor_cell_face_n
      {calculate_neighbor_face_n(current_cell, neighbor_cell, current_cell_face_n)};

    if (dim == 2)
      {
        Assert((current_cell->face(current_cell_face_n)->vertex(0)
                == neighbor_cell->face(neighbor_cell_face_n)->vertex(0)) !=
               (current_cell->face(current_cell_face_n)->vertex(1)
                == neighbor_cell->face(neighbor_cell_face_n)->vertex(1)),
               ExcMessage("Exactly one vertex should line up."));

        if (neighbor_cell->face(neighbor_cell_face_n)->vertex(0)
            == current_cell->face(current_cell_face_n)->vertex(0))
          {
            return 0;
          }
        else
          {
            return 1;
          }
      }
    if (dim == 3)
      {
        // needs a four-way xor statement in the Assert: exactly one of the
        // four vertices should line up.
      }
    return numbers::invalid_unsigned_int;
  }
}
