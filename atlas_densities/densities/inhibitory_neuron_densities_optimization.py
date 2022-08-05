"""Functions to create volumetric neuron densities for the following inhibitory neuron types of
the mouse brain:
- PV+
- SST+
- VIP+
- GAD67+ (a superset of the union of the previous three)

The algorithm is presented in "Atlas of inhibitory neurons in the mouse brain" by D. Rodarie et
al., to appear in <journal>.

This code supersedes the ``refined_inhibitory_neuron_densities`` module, which is maintained to
allow comparisons with the former approach of the same article.

The main function resolves a linear program which aims at minimizing the absolute differences
between output cell counts and input estimates while enforcing the consistency of cell count sums
across the brain region hierarchy.

The file ``doc/source/bbpp82_628_linear_program.pdf`` serves as a documentation of the linear
program, see Section 2 in particular. The initalization of the linear program relies on average
density estimates, obtained obtained for instance via the ``fitting`` module. We refer to these
estimates as the initial estimates.

The output are volumetric density nrrd files, one for each of the aforementioned inhibitory
neuron types. Volumetric densities are derived from the cell counts obtained as the solution of
the previous linear program.

A note on the linear program:

The variable ``x_result`` used in this module is a container for the output cell counts.
It encompasses the variables ``x_{i, m}`` defined in Section 2 of
``bbpp82_628_linear_program.pdf``. The variable ``x_result`` is initialized with the estimates
which are taken for certain (when the "confidence" numbers ``sigma_{i, m}`` are close to 0.0)
and with NaN otherwise. If ``x_result.at[(id_, marker)]`` is set to NaN, then a decision variable
``x_{i, m}`` is introduced where i = id_ and m = marker. Once the program is resolved,
``x_result.at[(id_, marker)]`` is set with the solution value of ``x_{i, m}``. Similarly, the
variable deltas used in this module corresponds to the decision variables ``delta_{r, m}`` of
the pdf file.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from atlas_commons.typing import AnnotationT, FloatArray
from scipy.optimize import linprog
from tqdm import tqdm
from voxcell import RegionMap

from atlas_densities.densities.inhibitory_neuron_densities_helper import (
    average_densities_to_cell_counts,
    check_region_counts_consistency,
    get_cell_types,
    replace_inf_with_none,
    resize_average_densities,
)
from atlas_densities.densities.utils import (
    compute_region_cell_counts,
    compute_region_volumes,
    get_hierarchy_info,
)
from atlas_densities.exceptions import AtlasDensitiesError, AtlasDensitiesWarning

L = logging.getLogger(__name__)

SKIP = np.inf  # Used to signify that a delta variable of the linear program should be removed
KEEP = np.nan  # Used to signify that a delta variable should be added to the linear program


def set_known_values(
    region_counts: pd.DataFrame,
    id_counts: pd.DataFrame,
    hierarchy_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the linear program variables `x_result` (cell counts) and `delta`
    (distances to initial estimates) and set the cell counts in `x_result` when those are known with
    certainty.

    The general rule is that if the standard variation associated to a cell count value from
    `region_counts` is (close to) 0.0, the output cell count is set with its initial estimate.
    This corresponds for instance to cell counts within regions known to contain only
    inhibitory neurons (the "known" overall neuron density can then be used) are known to
    contain no inhibitory cells, in which case subtype cell counts are set to 0.0.

    Args:
        region_counts: data frame returned by
            :fun:`densities.inhibitory_densities_helper.average_densities_to_cell_counts`.
            A region is understood as a region of the brain hierarchy and includes all descendant
            subregions.
        id_counts: data frame returned by
            :fun:`densities.inhibitory_densities_helper.average_densities_to_cell_counts` but
            indexed by unique integer region identifiers instead of region names.
            Cell counts are limited to 3D regions defined by a single integer identifier.
            Descendants subregions are therefore not included.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns: a pair (x_result, deltas) which is container for the variables of the linear program.
        x_result: data frame initialized with either cell count estimates, if they are knownn for
            certain (standard deviation close to 0.0), or with NaNs otherwise. It has one column
            for each cell type in `cell_types`. Its index is a sorted list of integer region
            identifiers, a. k. a., labels.
        deltas: data frame corresponding to the variables of the linear programs.
            These variables represent the absolute values of the differences between the cell
            count variables and their initial estimates. The linear program attempts to minimize a
            weighted sum of the `deltas`. The `deltas` data frame has one column for each cell
            type in `cell_types`. Its index is a list of region names (str).
            The data frame is initialized with zeros for regions where the cell count estimate is
            given for certain (standard deviation close to 0.0), with `SKIP` if no estimate
            is available and with `KEEP` otherwise. The latter case corresponds to a variable
            delta_{r, m} that is actually added to the linear program.
    """
    stddev_tolerance = 1e-2  # absolute tolerance used to compare standard deviation to 0.0
    cell_types = get_cell_types(region_counts)

    assert cell_types == get_cell_types(id_counts)
    assert np.all(hierarchy_info["brain_region"] == region_counts.index)
    assert np.all(hierarchy_info.index == id_counts.index)

    def _zero_descendants(id_, cell_type, x_result, deltas, hierarchy_info):
        """
        Zeroes the count of `cell_type` cells in every descendant regions of `id_`,
        including `id_`.
        """
        desc_ids = list(hierarchy_info.at[id_, "descendant_id_set"])
        x_result.loc[desc_ids, cell_type] = 0.0
        desc_names = hierarchy_info.loc[desc_ids, "brain_region"]
        deltas.loc[desc_names, cell_type] = 0.0

    L.info("Preparing variables layout ...")
    x_result = pd.DataFrame(
        {cell_type: np.full(len(id_counts), KEEP) for cell_type in cell_types},
        index=id_counts.index,  # indexed by integer identifiers
    )
    deltas = pd.DataFrame(
        {cell_type: np.full(len(region_counts), KEEP) for cell_type in cell_types},
        index=region_counts.index,  # indexed by region names (str)
    )

    L.info("Setting known values ...")
    # Set delta_{r, m} with `SKIP` if the corresponding cell count estimate is missing.
    # This means that the region r won't impose any constraint wrt to the marker m in the linear
    # program, i. e., the delta_{r, m} variable is omitted for such r and m.
    for (id_, region_name) in zip(hierarchy_info.index, region_counts.index):
        for cell_type in cell_types:
            # A region without cell count estimate for `cell_type` adds no constraint
            # to the linear program.
            if np.isnan(region_counts.at[region_name, cell_type + "_standard_deviation"]):
                deltas.loc[region_name, cell_type] = SKIP

        # If the initial cell count estimate of gad67+ (all inhibitory neurons) in `region_name`
        # and its standard deviation are both close to 0.0, we set the cell final counts of every
        # inhibitory neuron subtypes in every descendant regions to 0.0.
        if np.isclose(
            region_counts.at[region_name, "gad67+_standard_deviation"], 0.0, atol=stddev_tolerance
        ) and np.isclose(region_counts.at[region_name, "gad67+"], 0.0, atol=stddev_tolerance):
            for cell_type in cell_types:
                _zero_descendants(id_, cell_type, x_result, deltas, hierarchy_info)

    # Set the (possibly non-zero) cell count estimates which are given with certainty.
    for (id_, region_name) in zip(hierarchy_info.index, region_counts.index):
        for cell_type in cell_types:
            if np.isclose(
                region_counts.at[region_name, cell_type + "_standard_deviation"],
                0.0,
                atol=stddev_tolerance,
            ):
                if x_result.at[id_, cell_type] == 0.0 and id_counts.at[id_, cell_type] != 0.0:
                    raise AtlasDensitiesError(
                        f"The count of {cell_type} cells in the atomic 3D region with id {id_} "
                        f"(corresponding name: '{region_name}') has the non-zero estimate "
                        f"{id_counts.at[id_, cell_type]} which is given for certain whereas an "
                        f"ancestor region has a zero cell count."
                    )
                # Cell count estimates whose associated standard deviations are close to 0.0
                # are used as definitive estimates.
                x_result.at[id_, cell_type] = id_counts.at[id_, cell_type]
                deltas.at[region_name, cell_type] = 0.0

    return x_result, deltas


def create_bounds(
    x_result: pd.DataFrame,
    deltas: pd.DataFrame,
    neuron_counts: pd.DataFrame,
) -> Tuple[FloatArray, Dict[Tuple[int, str], int], Dict[Tuple[str, str], int]]:
    """
    Create the `bounds` array of shape (N, 2) where N is the number of variables
    of the linear program.

    The `bounds` array define a range constraining each variable.
    The upper bounds are given by the corresponding overall `neuron_counts`.

    Args:
        x_result: see return value of :fun:`set_known_values`. A `KEEP` entry means that a cell
            count variable is added to the program.
        deltas: see return value of :fun:`set_known_values`. A `KEEP` entry means that a
            delta variable is added to the program.
        neuron_counts: data frame with one column "cell_count" holding the neuron count each 3D
            region labeled by an integer identifier in `neuron_count.index`.

    Returns:
        a tuple (`bounds`, `x_map`, `deltas_map`) where
        - `bounds` is a float array of shape (N, 2) with N the total number of decision variables.
            ``np.inf`` values in `bounds[..., 1]` indicates that there is no constraining upper
            bound.
        - `x_map` is a dict mapping pairs (id_, cell_type) to indices of `bounds`.
            The keys of this dict corresponds to cell count variables in the linear program.
        - `deltas_map` is a dict mapping pairs (region_name, cell_type) to to indices of `bounds`.
            The keys of this dict corresponds to the slack "delta" variables which represent
            distances of cell count variables to initial estimates.
    """
    x_map = {}
    deltas_map = {}
    bounds = []  # list of pairs (lower_bound, upper_bound)
    cell_types = get_cell_types(x_result)
    for id_ in x_result.index:
        for cell_type in cell_types:
            if np.isnan(x_result.at[id_, cell_type]):
                assert neuron_counts.at[id_, "cell_count"] >= 0.0
                bounds.append((0.0, neuron_counts.at[id_, "cell_count"]))
                x_map[(id_, cell_type)] = len(bounds) - 1

    for region_name in deltas.index:
        for cell_type in cell_types:
            if np.isnan(deltas.at[region_name, cell_type]):
                bounds.append((0.0, np.inf))
                deltas_map[(region_name, cell_type)] = len(bounds) - 1

    return np.asarray(bounds, dtype=float), x_map, deltas_map


def create_aub_and_bub(
    # pylint: disable=too-many-locals
    x_result: pd.DataFrame,
    region_counts: pd.DataFrame,
    x_map: Dict[Tuple[int, str], int],
    deltas_map: Dict[Tuple[str, str], int],
    hierarchy_info: pd.DataFrame,
) -> Tuple[FloatArray, FloatArray]:
    """
    Create the matrix and the right-hand side vector of the linear inequality constraints.

    Linear inequality constraints are of two kinds:
    - a pair of constraints expressing the fact that the distances to the initial estimates
        is less or equal to a delta variable (the program minimizes a weighted sum of delta
        variables)
    - an inequality expressing the fact the cell count of gad67+, the total of inhibitory
        neuron counts, is at least the sum of every inhibitory subtype cell counts.

    Args:
        x_result: see return value of :fun:`set_known_values`. A `KEEP` entry means that a
            cell count variable is added to the program.
        region_counts: data frame with one column for per cell type and with index
            a list of region names. This data frame holds the initial cell count estimates.
        x_map: dict mapping pairs (id_, cell_type) to integer indices of the global list of
            variables.
        deltas_map: dict mapping pairs (region_name, cell_type) to integer indices of the global
            list of variables.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns: a pair (`a_ub`, `b_ub`) where `a_ub` is a float array of shape (C, N) and
        `b_ub` is a float array of shape (C,). The integer N stands for the number of variables
        while C stands for the number of inequality constraints.
        The implemented constraints are: a_ub X <= b_ub where X stands for the cell count
        variables.
    """

    def check_constraint_3c(b_value: float, region_name: str, id_: int) -> None:
        """
        Check if `b_value` is negative in the right-handside of an inequality of type (3c).

        Warns: issues an  AtlasDensitiesWarning if `b_value` is negative, as
            a potential issue is hence detected.
        """
        if b_value < 0.0:
            warnings.warn(
                f"The inequality constraint of type (3c) for r = '{region_name}', "
                f"id = {id_} has the negative right-handside value {b_value}, which will "
                f"cause the corresponding delta variable to be at least {np.abs(b_value)} "
                f"whereas the target delta value is 0.0.",
                AtlasDensitiesWarning,
            )

    def check_constraint_3e(
        constraint: FloatArray, b_value: float, region_name: str, id_: int
    ) -> bool:
        """
        Check if `contraint` is a valid inequality constraint of type (3e).

        Returns:
            True if the constraint is not void.

        Raises:
           AtlasDensitiesWarning if an inconsistency is detected.
        """
        if b_value != 0.0 or np.any(constraint != 0.0):  # Do not add void constraints
            if b_value < 0.0 and np.all(constraint >= 0.0):
                raise AtlasDensitiesError(
                    f"Inconsistent inequality for constraint of type (3e): "
                    f"r = '{region_name}', id = {id_}.\n The left-hand side is zero whereas "
                    f"the right-hand side is negative."
                )
            if b_value < 0.0:
                warnings.warn(
                    f"The inequality constraint of type (3e) for id = {id_} "
                    f"has the negative right-handside value {b_value}.",
                    AtlasDensitiesWarning,
                )
            return True

        return False

    a_ub = []
    b_ub = []
    variable_count = len(x_map) + len(deltas_map)
    cell_types = get_cell_types(region_counts)
    for (id_, region_name, set_) in zip(
        hierarchy_info.index, hierarchy_info["brain_region"], hierarchy_info["descendant_id_set"]
    ):
        for cell_type in cell_types:
            if (region_name, cell_type) in deltas_map:
                constraint = np.zeros((variable_count,), dtype=float)
                b_value = region_counts.at[region_name, cell_type]
                for desc_id in set_:
                    if (desc_id, cell_type) in x_map:
                        constraint[x_map[(desc_id, cell_type)]] = 1.0
                    else:
                        b_value -= x_result.at[desc_id, cell_type]
                constraint[deltas_map[(region_name, cell_type)]] = -1.0
                a_ub.append(constraint)  # Inequality (3c) from the pdf file
                check_constraint_3c(b_value, region_name, id_)
                b_ub.append(b_value)

                constraint_minus = -constraint
                constraint_minus[deltas_map[(region_name, cell_type)]] = -1.0
                a_ub.append(constraint_minus)  # Inequality (3d) from the pdf file
                b_ub.append(-b_value)

        constraint = np.zeros((variable_count,), dtype=float)
        b_value = 0.0
        if (id_, "gad67+") not in x_map:
            b_value += x_result.at[id_, "gad67+"]
        else:
            constraint[x_map[(id_, "gad67+")]] = -1.0
        for cell_type in set(cell_types) - {"gad67+"}:
            if (id_, cell_type) in x_map:
                constraint[x_map[(id_, cell_type)]] = 1.0
            else:
                b_value -= x_result.at[id_, cell_type]
        if check_constraint_3e(constraint, b_value, region_name, id_):
            a_ub.append(constraint)  # Inequality (3e) from the pdf file
            b_ub.append(b_value)

    return np.asarray(a_ub, dtype=float), np.asarray(b_ub, dtype=float)


def create_volumetric_densities(
    x_result: pd.DataFrame,
    annotation: AnnotationT,
    neuron_density: FloatArray,
    neuron_counts: pd.DataFrame,
    cell_types: List[str],
) -> Dict[str, FloatArray]:
    """
    Create volumetric densities for each cell type in `cell_types` based on the `x_result`cell
    counts.

    Args:
        x_result: see return value of :fun:`set_known_values`. A `KEEP` entry means that a cell
            count variable is added to the program.
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        neuron_density: non-negative float array of shape (W, H, D). This array holds the
            volumetric neuron density of the brain model expressed in number of neurons per mm^3.
        neuron_counts: data frame with one column "cell_count" holding the neuron count each 3D
            region labeled by an integer identifier in `neuron_count.index`.
        cell_types:
            list of cell type names, e.g., ["gad67+", "pv+", "sst+", "vip+"].

    Returns:
        dict whose keys are cell types (str) and whose values a float array of shape
        `annotation.shape` (which is assumed to coincide with `neuron_density.shape`). The value
        of a voxel in each array represents a cell density expressed in number of cells per mm^3.
    """

    densities = {cell_type: neuron_density.copy() for cell_type in cell_types}
    for id_ in tqdm(neuron_counts.index):
        mask = annotation == id_
        neuron_count = neuron_counts.at[id_, "cell_count"]
        for cell_type in cell_types:
            if np.isclose(neuron_count, 0.0):
                densities[cell_type][mask] = 0.0
            else:
                densities[cell_type][mask] *= x_result.at[id_, cell_type] / neuron_count

    return densities


def _compute_initial_cell_counts(
    annotation: AnnotationT,
    voxel_volume: float,
    average_densities: pd.DataFrame,
    hierarchy_info: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute initial cell counts for various cell types based on `average_densities`.

    Args:
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        voxel_volume: volume expressed in mm^3 of a voxel. The input volumetric data (annotation
            and neuron density) are assumed to have voxels of the same dimensions.
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.densities.fitting.linear_fitting` containing the average
            cell densities of brain regions and their associated standard deviations. Columns are
            labelled by T and T_standard_deviation for various cell types T. The index of
            `average_densities` is a list of region names.
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.

    Returns:
        a pair of data frames (`region_counts`, `id_counts`).
        - `region_counts` has the return value format of
            :fun:`densities.inhibitory_densities_helper.average_densities_to_cell_counts`.
            It holds region cell counts as well as associated standard deviations.
        - `id_counts` has the return value format of
            :fun:`densities.inhbitory_densities_helper.average_densities_to_cell_counts` except
            that its index is a list of unique integer region identifiers.
            It holds the cell counts of the 3D regions labeled by a unique identifier, as well as
            associated standard deviations (descendant subregions are excluded).
    """
    L.info("Computing the volume of every 3D region ...")
    volumes = compute_region_volumes(annotation, voxel_volume, hierarchy_info)

    L.info("Computing cell count estimates in every 3D region ...")
    region_counts = average_densities_to_cell_counts(average_densities, volumes)

    # Detect cell counts inconsistency between a region and its descendants when
    # estimates are deemed as certain.
    check_region_counts_consistency(region_counts, hierarchy_info)

    L.info("Computing cell count estimates in atomic 3D region ...")
    volumes.drop("volume", axis=1, inplace=True)
    volumes.rename(columns={"id_volume": "volume"}, inplace=True)
    id_counts = average_densities_to_cell_counts(average_densities, volumes)
    id_counts.index = volumes.index  # unique integer identifiers

    return region_counts, id_counts


def _check_variables_consistency(
    x_result: pd.DataFrame,
    deltas: pd.DataFrame,
    cell_types: List[str],
    neuron_counts: pd.DataFrame,
    hierarchy_info: pd.DataFrame,
) -> None:
    """
    Raises an error if a delta variable indexed by (region_name, cell_type) is set with a non-NaN
    value whereas an x_result variable indexed by (desc_id, cell_type) is set with a NaN value for
    some descendant id of region_name.

    Raises also an error if a cell count value marked as final in `x_result` exceeds its
    prescribed upper bound from `neuron_counts`.

    Args:
        x_result: see return value of :fun:`set_known_values`.
        deltas: idem
        cell_types: list of cell type names, e.g., ["pv+", "sst+", "vip+", "gad67+"].
        hierarchy_info: data frame returned by
            :func:`atlas_densities.densities.utils.get_hierarchy_info`.
        neuron_counts: data frame with one column "cell_count" holding the neuron count each 3D
            region labeled by an integer identifier in `neuron_count.index`.

    Raises:
        AtlasDensitiesError if on the the following assumptions is violated:
        - if cell count estimate of a region is known with certainty for a given cell type,
        then the cell count of every descendant region is also known with certainty.
        - a cell count estimate which is given for certain does not
    """
    cell_count_tolerance = 1e-2  # absolute tolerance to rule out round-off errors
    for region_name, id_, id_set in zip(
        deltas.index, hierarchy_info.index, hierarchy_info["descendant_id_set"]
    ):
        for cell_type in cell_types:
            if np.isfinite(deltas.loc[region_name, cell_type]):
                for desc_id in id_set:
                    if np.isnan(x_result.loc[desc_id, cell_type]):
                        raise AtlasDensitiesError(
                            f"Cell count estimate of region named '{region_name}' for cell type "
                            f"{cell_type} was given for certain whereas the cell count of "
                            f"descendant id {desc_id} is not certain."
                        )
            neuron_count = neuron_counts.loc[id_, "cell_count"]
            if (
                not np.isnan(x_result.loc[id_, cell_type])
                and x_result.loc[id_, cell_type] > neuron_count
            ):
                if x_result.loc[id_, cell_type] <= neuron_count + cell_count_tolerance:
                    x_result.loc[id_, cell_type] = neuron_count
                else:
                    raise AtlasDensitiesError(
                        f"Cell count estimate of atomic 3D region with id {id_} for cell type"
                        f" {cell_type} is {x_result.loc[id_, cell_type]}, which exceeds the "
                        f"estimated overall neuron count for this region, that is {neuron_count}."
                    )


def _check_linprog_consistency(a_ub: FloatArray, b_ub: FloatArray, bounds: FloatArray) -> None:
    """
    Check some basic expectations on the linear program providing the cell count estimates.

    Checks are based the program description in Section 2 of
    ``doc/source/bbpp82_628_linear_program.pdf``

    Args:
        a_ub: see :fun:`create_aub_and_bub` output description.
        b_ub: see :fun:`create_aub_and_bub` output description.
        bounds: see :fun:`create_bounds` output description.

    Raises:
        AtlasDensitiesError if some inconsistency with the description of
        ``doc/source/bbpp82_628_linear_program.pdf`` is detected.
    """
    if not np.all(np.isfinite(a_ub)):
        raise AtlasDensitiesError(
            "Unexpected infinite values in the matrix A_ub encoding inequality constraints."
        )
    if not np.all(np.isfinite(b_ub)):
        raise AtlasDensitiesError("Unexpected infinite values in b_ub.")
    diff = set(np.unique(a_ub)) - {-1.0, 0.0, 1.0}
    if diff:
        raise AtlasDensitiesError(
            f"Unexpected coefficient values the matrix A_ub "
            f"encoding inequality constraints:  {diff}"
        )
    if len(bounds) > 0:  # `bounds` is empty if all values are given for certain.
        if not np.all(bounds[:, 0] == 0.0):
            raise AtlasDensitiesError("Unexpected non-zero lower bounds in b_ub.")
        if not np.all(bounds[:, 1] >= 0.0):
            raise AtlasDensitiesError("Unexpected negative upper bounds in b_ub.")


def create_inhibitory_neuron_densities(  # pylint: disable=too-many-locals
    hierarchy: dict,
    annotation: AnnotationT,
    voxel_volume: float,
    neuron_density: FloatArray,
    average_densities: pd.DataFrame,
) -> Dict[str, FloatArray]:
    """
    Create a 3D float array for each cell type that labels a column of `average_densities`.

    Voxel values are cell density values expressed in number of cells per mm^3.
    For a given `annotation` array, a non-zero annotated voxel belongs to an "atomic" 3D region
    whose label is either a leaf of the brain region hierarchy tree or, more surprisingly, not a
    leaf when annotation failed to be more precise. See
    https://community.brain-map.org/t/how-to-interpret-3d-regions-which-are-labelled-by-non-leaf-identifiers-of-1-json/1159
    for discussion on AIBS annotation issues.

    This function computes new cell count estimates based on the initial estimates of
    `average_densities`. This is done by solving a linear program which enforces cell counts
    consistency:
    - the inhibitory neuron count in a region is at least the sum of counts of inhibitory cell
        subtypes under consideration;
    - each neuron subtype count does not exceed the overall cell count prescribed by
        `neuron_density`.

    The linear program minimizes the distances of each cell count to its initial
    estimate when it exists while enforcing the aforementioned consistency.
    The objective function is the sum of the distances to available region cell count estimates
    weighted by the inverses of the associated standard deviations.

    See "Atlas of inhibitory neurons in the mouse brain" by D. Rodarie et al., to appear in
    <journal> for a detailed description of the algorithm.

    Note:
        The density values of regions which are not listed in `average_densities` are
        set to 0.0. Idem for regions whose initial estimates contain a NaN value.

    Args:
        hierarchy: dict holding the tree of the brain regions hierarchy. Children regions
            are accessed via the "children" attribute. (dict counterpart of 1.json or
            hierarchy.json)
        annotation: int array of shape (W, H, D) holding the annotation of the whole
            brain model. (The integers W, H and D are the dimensions of the array).
        voxel_volume: volume of a voxel expressed in mm^3. The input volumetric data (annotation
            and neuron density) are assumed to have voxels of the same dimensions.
        neuron_density: non-negative float array of shape (W, H, D). This array holds the
            volumetric neuron density of the brain model expressed in number of neurons per mm^3.
        average_densities: a data frame whose columns are described in
            :func:`atlas_densities.densities.fitting.linear_fitting` containing the average
            cell densities of brain regions and their associated standard deviations. Columns are
            labelled by T and T_standard_deviation for various cell types T. The index of
            `average_densities` is a list of region names.

    Returns:
        a dict whose keys are cell types (e.g., "gad67+", "pv+", "sst+" and "vip+") and whose
        values are float arrays of shape (W, H, D) holding the cell density of each cell type
        expressed in number of cells per mm^3.

    Raises:
        AtlasDensitiesError if some inconsistency in the input data has been detected or if the
        linear program cannot be solved.
    """

    hierarchy_info = get_hierarchy_info(RegionMap.from_dict(hierarchy), root="root")
    average_densities = resize_average_densities(average_densities, hierarchy_info)

    L.info("Initialization of the linear program: started")
    region_counts, id_counts = _compute_initial_cell_counts(
        annotation, voxel_volume, average_densities, hierarchy_info
    )

    L.info("Retrieving overall neuron counts in atomic 3D regions ...")
    neuron_counts = compute_region_cell_counts(
        annotation, neuron_density, voxel_volume, hierarchy_info, with_descendants=False
    )
    assert np.all(neuron_counts["cell_count"] >= 0.0)

    L.info("Setting the known values ...")
    x_result, deltas = set_known_values(region_counts, id_counts, hierarchy_info)
    # We assume that if the cell count of `cell_type` in `region_name` is known with certainty
    # then the same holds in every descendant subregion.
    _check_variables_consistency(
        x_result, deltas, get_cell_types(region_counts), neuron_counts, hierarchy_info
    )

    L.info("Setting variable bounds and further inequality constraints ...")
    bounds, x_map, deltas_map = create_bounds(x_result, deltas, neuron_counts)
    variable_count = len(x_map) + len(deltas_map)
    assert set(x_map.values()) == set(range(len(x_map)))
    assert set(deltas_map.values()) == set(range(len(x_map), variable_count))

    a_ub, b_ub = create_aub_and_bub(x_result, region_counts, x_map, deltas_map, hierarchy_info)

    assert variable_count == len(bounds)
    assert variable_count == a_ub.shape[1]
    _check_linprog_consistency(a_ub, b_ub, bounds)

    L.info("Initialization of the linear program: finished.")
    if variable_count != 0:  # linprog raises a ValueError if c_row is empty
        c_row = np.zeros((variable_count,), dtype=float)
        for (region_name, cell_type), index in deltas_map.items():
            std_name = cell_type + "_standard_deviation"
            c_row[index] = 1.0 / region_counts.at[region_name, std_name]
            assert c_row[index] > 0.0

        L.info(
            "Solving linear program with %d variables and %d inequality constraints",
            variable_count,
            len(b_ub),
        )
        result = linprog(
            c=c_row,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=replace_inf_with_none(bounds),
            method="highs",
        )
        if not result.success:
            raise AtlasDensitiesError(
                "The linear program minimizing the distances to cell count estimates couldn't "
                "be solved."
            )

        # inhibitory neuron count estimates x_{i, m} and delta_{r, m} values are non-negative.
        # Due to float rounding errors, we set a minimum negative value epsilon below which we
        # consider a negative value to be an error. x_ and delta_ being neuron counts derived
        # values, epsilon does not have to be very small (i.e., < -1e-10).
        epsilon = -1e-9
        assert np.all(result.x >= epsilon)
        result.x[result.x < 0.0] = 0.0
        # inhibitory neuron count estimates x_{i, m} don't exceed prescribed bounds
        # (over all neuron counts)
        assert np.all(result.x <= bounds[:, 1])

        L.info("Mapping linear program output back to cell count variables ...")
        for (id_, cell_type), row_index in x_map.items():
            x_result.at[id_, cell_type] = result.x[row_index]

    L.info("Creating volumetric densities out of cell count estimates ...")

    return create_volumetric_densities(
        x_result, annotation, neuron_density, neuron_counts, get_cell_types(region_counts)
    )
