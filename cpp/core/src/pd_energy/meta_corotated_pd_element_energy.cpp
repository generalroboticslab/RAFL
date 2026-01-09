#include "pd_energy/meta_corotated_pd_element_energy.h"
#include "common/geometry.h"

template<int dim>
const Eigen::Matrix<real, dim, dim> MetaCorotatedPdElementEnergy<dim>::ProjectToManifold(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return R;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> MetaCorotatedPdElementEnergy<dim>::ProjectToManifoldDifferential(
    const Eigen::Matrix<real, dim, dim>& F, const Eigen::Matrix<real, dim, dim>& dF) const {
    Eigen::Matrix<real, dim, dim> R, S;
    PolarDecomposition(F, R, S);
    return dRFromdF(F, R, S, dF);
}

template<int dim>
const Eigen::Matrix<real, dim, dim> MetaCorotatedPdElementEnergy<dim>::ProjectToManifold(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary) const {
    return F_auxiliary.R();
}

template<int dim>
const Eigen::Matrix<real, dim, dim> MetaCorotatedPdElementEnergy<dim>::ProjectToManifoldDifferential(
    const DeformationGradientAuxiliaryData<dim>& F_auxiliary, const Eigen::Matrix<real, dim, dim>& projection,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    return dRFromdF(F_auxiliary.F(), F_auxiliary.R(), F_auxiliary.S(), dF);
}

template class MetaCorotatedPdElementEnergy<2>;
template class MetaCorotatedPdElementEnergy<3>;