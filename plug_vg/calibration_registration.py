"""Independent held-plug calibration: observations and endpoint-anchored CAD fit.

No GT inputs and no dependency on the grasp registration algorithms.
"""
from __future__ import annotations

import numpy as np

from .visible_points import extract_visible_points_from_mask


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if vector.shape != (3,) or not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("Axis must be a finite, nonzero 3-vector")
    return vector / norm


def calibration_priors(t_end_camera, t_base_end, clamp_center_end=(0., 0., .420)):
    """Return clamp center and base +Z in the RGB camera frame, in meters."""
    t_camera_end = np.linalg.inv(t_end_camera)
    center = (t_camera_end @ np.r_[clamp_center_end, 1.])[:3]
    up = np.linalg.solve((t_base_end @ t_end_camera)[:3, :3], [0., 0., 1.])
    return center, _unit(up)


def calibration_observation(image, depth, camera, prediction):
    """Preserve the highest-confidence detection's actual head/tail class."""
    if prediction.masks is None or len(prediction.boxes) == 0:
        raise ValueError("segmentation_missing")
    index = int(prediction.boxes.conf.argmax())
    class_id = int(prediction.boxes.cls[index])
    label = prediction.names[class_id]
    if label not in ("plug_head", "plug_tail"):
        raise ValueError(f"unsupported_calibration_class: {label}")
    polygon = prediction.masks.xy[index].tolist()
    clouds = []
    for erosion, voxel in ((5, .004), (0, 0.)):
        extraction = extract_visible_points_from_mask(
            image, depth, camera, polygon, min_depth_m=.1, max_depth_m=1.,
            voxel_size_m=voxel, min_points=200, mask_erosion_px=erosion,
            mad_z_threshold=0.)
        if extraction.status != "ok":
            raise ValueError(extraction.reason)
        clouds.append(extraction)
    return dict(class_name=label, class_id=class_id,
                confidence=float(prediction.boxes.conf[index]), polygon_xy=polygon,
                core=clouds[0], raw=clouds[1])


def endpoint_on_axis(points, center, axis, class_name):
    """Estimate an axial boundary, not the centroid of a visible end face."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Need at least three finite boundary points")
    if class_name not in ("plug_head", "plug_tail"):
        raise ValueError("Expected plug_head or plug_tail")
    axis = _unit(axis)
    quantile = .02 if class_name == "plug_tail" else .98
    projections = (points - center) @ axis
    offset = float(np.quantile(projections, quantile))
    return dict(axis_camera=axis, endpoint_camera_m=np.asarray(center) + offset * axis,
                endpoint_offset_m=offset, quantile=quantile,
                support_mask=np.abs(projections - offset) <= .003)


def initial_endpoint_candidates(core, raw, center, up, class_name):
    """Two unranked seeds; the later CAD stage must resolve surface bias."""
    core = np.asarray(core, dtype=float)
    if core.ndim != 2 or core.shape[1] != 3 or len(core) < 3 or not np.isfinite(core).all():
        raise ValueError("Need at least three finite core points")
    _, _, vt = np.linalg.svd(core - core.mean(axis=0), full_matrices=False)
    radial = np.median(core, axis=0) - center
    if class_name == "plug_tail":
        radial = -radial
    seeds = [("pca", vt[0]), ("clamp_to_observation", radial)]
    candidates = []
    for name, axis in seeds:
        axis = _unit(axis)
        if axis @ up < 0:
            axis = -axis
        candidates.append(dict(name=name, **endpoint_on_axis(raw, center, axis, class_name)))
    return candidates


def filter_calibration_clouds(core, raw):
    """Keep the largest spatial component; retain raw boundary near that surface.

    Distances account for the existing 4 mm core voxel grid and mask erosion.
    Return masks so all discarded depth remains available for visual review.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(core))
    labels = np.asarray(cloud.cluster_dbscan(eps=.012, min_points=5))
    values, counts = np.unique(labels[labels >= 0], return_counts=True)
    if not len(values):
        raise ValueError("no_connected_calibration_surface")
    core_keep = labels == values[np.argmax(counts)]
    raw_keep = cKDTree(np.asarray(core)[core_keep]).query(raw)[0] <= .008
    if core_keep.sum() < 20 or raw_keep.sum() < 20:
        raise ValueError("insufficient_connected_calibration_surface")
    return core_keep, raw_keep


_candidate_worker_solver = None


def _initialize_candidate_worker(vertices, triangles, tail_x, head_x):
    import open3d as o3d
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    global _candidate_worker_solver
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    _candidate_worker_solver = EndpointCADRegistration(mesh, tail_x, head_x)


def _candidate_worker_ready(_):
    return _candidate_worker_solver is not None


def _fit_candidate_subset(payload):
    return _candidate_worker_solver.fit_coarse(*payload[:5], task_indices=payload[5])['candidates']


class EndpointCADRegistration:
    """Cached unsigned CAD surface queries; stage-2 orientation-only fit.

    The axis passes through the clamp. Translation follows the observed axial
    boundary for every orientation, leaving three rotation variables including
    auxiliary roll. The output transform maps CAD grasp coordinates to camera.
    """

    def __init__(self, mesh, tail_x=-.0778, head_x=.09, *, parallel=False):
        import open3d as o3d

        self.candidate_pool = None
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.tail_x, self.head_x = float(tail_x), float(head_x)
        if parallel:
            from concurrent.futures import ProcessPoolExecutor
            from multiprocessing import get_context
            self.candidate_pool = ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn'),
                initializer=_initialize_candidate_worker,
                initargs=(np.asarray(mesh.vertices), np.asarray(mesh.triangles), tail_x, head_x))
            list(self.candidate_pool.map(_candidate_worker_ready, range(4)))

    def surface_distance(self, points):
        import open3d as o3d

        return self.scene.compute_distance(
            o3d.core.Tensor(np.asarray(points, dtype=np.float32)), nthreads=1).numpy().astype(float)

    def refine_icp(self, core, coarse_transform, center, class_name):
        """Robust observation-to-CAD point-to-plane ICP with two soft anchors.

        Missing matches pay the 10 mm cutoff cost, normalized by all observations.
        Closest surface normals are queried on the non-watertight CAD directly.
        A failed result retains coarse evidence but supplies no final transform.
        """
        import open3d as o3d
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        if class_name not in ("plug_head", "plug_tail"):
            raise ValueError("Expected plug_head or plug_tail")
        points = np.asarray(core, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20 or not np.isfinite(points).all():
            return dict(status="failed", reason="insufficient_finite_icp_points")
        points = points[np.linspace(0, len(points)-1, min(2000, len(points)), dtype=int)]
        coarse = np.asarray(coarse_transform, dtype=float)
        current = coarse.copy()
        endpoint_x = self.tail_x if class_name == "plug_tail" else self.head_x
        anchor = coarse[:3, 3] + endpoint_x*coarse[:3, 0]
        missing_cost = np.log1p((.010/.003)**2)

        def priors(transform):
            axis, translation = transform[:3, 0], transform[:3, 3]
            offset = np.asarray(center)-translation
            return np.r_[(offset-axis*(offset @ axis))/.005,
                         (translation+endpoint_x*axis-anchor)/.003]

        def correspondences(transform):
            local = (points-transform[:3, 3]) @ transform[:3, :3]
            query = self.scene.compute_closest_points(o3d.core.Tensor(local.astype(np.float32)), nthreads=1)
            closest, normals = query['points'].numpy().astype(float), query['primitive_normals'].numpy().astype(float)
            distance = np.linalg.norm(local-closest, axis=1)
            keep = (distance <= .010) & np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > .5)
            residual = np.sum((local-closest)*normals, axis=1)/.003
            cost = (np.log1p(residual[keep]**2).sum() + (~keep).sum()*missing_cost)/len(points) + np.sum(priors(transform)**2)
            return closest, normals, keep, float(cost)

        history = []
        stop_reason = "iteration_limit"
        for iteration in range(20):
            closest, normals, keep, cost = correspondences(current)
            if keep.sum() < 20:
                return dict(status="failed", reason="insufficient_icp_correspondences", history=history,
                            coarse_t_camera_grasp=coarse)

            def updated(delta):
                transform = current.copy()
                transform[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ current[:3, :3]
                transform[:3, 3] += delta[3:]
                return transform

            def residuals(delta):
                transform = updated(delta)
                local = (points[keep]-transform[:3, 3]) @ transform[:3, :3]
                plane = np.sum((local-closest[keep])*normals[keep], axis=1)/.003
                robust = np.sign(plane)*np.sqrt(np.log1p(plane**2))/np.sqrt(len(points))
                return np.r_[robust, priors(transform)]

            step = least_squares(residuals, np.zeros(6), x_scale=[.1,.1,.1,.01,.01,.01], max_nfev=8)
            if not np.isfinite(step.x).all():
                return dict(status="failed", reason="nonfinite_icp_update", history=history, coarse_t_camera_grasp=coarse)
            accepted = False
            for factor in (1., .5, .25, .125):
                trial = updated(step.x*factor)
                _, _, trial_keep, trial_cost = correspondences(trial)
                if trial_keep.sum() >= 20 and trial_cost <= cost:
                    accepted = True
                    break
            history.append(dict(iteration=iteration+1, cost_before=cost, cost_after=trial_cost if accepted else cost,
                                correspondences=int(keep.sum()), accepted=accepted))
            if not accepted:
                stop_reason = "no_descent_step"
                break
            current = trial
            if cost-trial_cost < 1e-6:
                stop_reason = "objective_converged"
                break
        _, _, keep, final_cost = correspondences(current)
        axis, translation = current[:3, 0], current[:3, 3]
        distance = self.surface_distance((points-translation) @ current[:3, :3])
        return dict(status="awaiting_visual_review", t_camera_grasp=current, history=history,
                    stop_reason=stop_reason, cost=final_cost, correspondences=int(keep.sum()),
                    surface_median_m=float(np.median(distance)), surface_p95_m=float(np.quantile(distance,.95)),
                    surface_within_10mm_fraction=float(np.mean(distance <= .010)),
                    clamp_axis_distance_m=float(np.linalg.norm(np.cross(center-translation, axis))),
                    endpoint_displacement_m=float(np.linalg.norm(translation+endpoint_x*axis-anchor)),
                    translation_change_m=float(np.linalg.norm(translation-coarse[:3,3])),
                    axis_change_deg=float(np.rad2deg(np.arccos(np.clip(axis @ coarse[:3,0],-1,1)))),
                    rotation_change_deg=float(np.rad2deg(Rotation.from_matrix(current[:3,:3] @ coarse[:3,:3].T).magnitude())))

    def fit_coarse(self, core, raw, center, up, class_name, *, task_indices=None):
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation

        seeds = initial_endpoint_candidates(core, raw, center, up, class_name)
        # Deterministic spatially voxelized core subset; preserve all raw endpoint pixels.
        points = np.asarray(core)[np.linspace(0, len(core)-1, min(512, len(core)), dtype=int)]
        endpoint_x = self.tail_x if class_name == "plug_tail" else self.head_x
        raw_centered = np.asarray(raw) - center
        boundary_cache = {}

        def boundary(axis):
            direction = _unit(axis)
            key = direction.tobytes()
            if key not in boundary_cache:
                offset = float(np.quantile(raw_centered @ direction, .02 if class_name == 'plug_tail' else .98))
                boundary_cache[key] = np.asarray(center) + offset*direction
            return boundary_cache[key]

        def optimize_candidate(task):
            seed, roll = task
            axis = seed["axis_camera"]
            helper = np.eye(3)[np.argmin(np.abs(axis))]
            y = _unit(np.cross(helper, axis))
            base = np.column_stack([axis, y, np.cross(axis, y)])
            rotation0 = base @ Rotation.from_rotvec([roll, 0., 0.]).as_matrix()

            def pose(delta):
                rotation = Rotation.from_rotvec(delta).as_matrix() @ rotation0
                end = {'endpoint_camera_m': boundary(rotation[:, 0])}
                translation = end["endpoint_camera_m"] - endpoint_x*rotation[:, 0]
                return rotation, translation, end

            evaluations = 0
            best_delta, best_cost = np.zeros(3), float("inf")

            class EvaluationBudgetReached(Exception):
                pass

            def objective(delta):
                nonlocal evaluations, best_delta, best_cost
                if evaluations >= 30:
                    raise EvaluationBudgetReached
                evaluations += 1
                rotation, translation, _ = pose(delta)
                if rotation[:, 0] @ up < 0:
                    return 100.
                distance = self.surface_distance((points-translation) @ rotation)
                cost = float(np.mean(np.log1p((distance/.003)**2)))
                if cost < best_cost:
                    best_delta, best_cost = np.array(delta), cost
                return cost

            # Three variables, finite differences; maxfun bounds actual surface calls.
            try:
                result = minimize(objective, np.zeros(3), method="L-BFGS-B",
                                  options=dict(maxfun=30, maxiter=30, eps=1e-4, ftol=1e-7))
                success, message = bool(result.success), str(result.message)
            except EvaluationBudgetReached:
                success, message = False, "30 objective evaluations reached; best evaluated pose retained"
            if not np.isfinite(best_cost):
                return None
            rotation, translation, end = pose(best_delta)
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, translation
            return dict(seed=seed["name"], roll_seed_rad=float(roll),
                                   cost=best_cost, t_camera_grasp=transform,
                                   endpoint_camera_m=end["endpoint_camera_m"],
                                   optimizer_success=success, optimizer_message=message,
                                   evaluations=evaluations)

        tasks = [(seed, roll) for seed in seeds for roll in np.arange(4)*np.pi/2]
        if self.candidate_pool is not None:
            payloads = [(core, raw, center, up, class_name, [i, i+4]) for i in range(4)]
            candidates = [candidate for group in self.candidate_pool.map(_fit_candidate_subset, payloads) for candidate in group]
            candidates.sort(key=lambda candidate: (0 if candidate['seed']=='pca' else 1, candidate['roll_seed_rad']))
        else:
            selected = tasks if task_indices is None else [tasks[i] for i in task_indices]
            candidates = [candidate for task in selected if (candidate := optimize_candidate(task)) is not None]
        candidates.sort(key=lambda item: item["cost"])
        if not candidates:
            raise ValueError("no_finite_calibration_candidate")
        best = candidates[0]
        distances = self.surface_distance((points-best["t_camera_grasp"][:3, 3]) @ best["t_camera_grasp"][:3, :3])
        return dict(status="awaiting_visual_review", **best, candidates=candidates,
                    surface_median_m=float(np.median(distances)), surface_p95_m=float(np.quantile(distances, .95)),
                    surface_within_10mm_fraction=float(np.mean(distances <= .010)))
