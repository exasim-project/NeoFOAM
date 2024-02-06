// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

namespace NeoFOAM {

/* A GeometricField represents a container for physical field data with
 *additional boundary which is assosiated with some geometric mesh.
 **
 */
template <class MeshType> class GeometricField : public IOObject {

private:
  // TODO could this be an ABC for access of geometry related functions
  std::shared_ptr<MeshType> mesh_;

  // NOTE container to abstract actual memory allocation and deallocation
  // away
  std::shared_ptr<device_array<scalar>> field_data_;

  /* Container for data on interfaces like processor, cyclic, AMI etc boundaries
   */
  std::shared_ptr<device_array<scalar>> interface_data_;

public:
  /*constructor*/

  /*arithmetic operator*/
  GeometricField<MeshType> operator+(const GeometricField<MeshType> &b) const;

  GeometricField<MeshType> operator-(const GeometricField<MeshType> &b) const;

  /* piecewise multiplication of two fields */
  GeometricField<MeshType> operator*(const GeometricField<MeshType> &b) const;

  /* scaling of a field by a scalar */
  GeometricField<MeshType> operator*(const scalar &b) const;

  /* scaling of a field by a scalar */
  GeometricField<MeshType> operator/(const scalar &b) const;
};

} // namespace NeoFOAM
