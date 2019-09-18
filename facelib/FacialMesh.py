import os
import cv2
# noinspection PyUnresolvedReferences
import eos
import numpy as np

"""
This app demonstrates estimation of the camera and fitting of the shape
model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
In addition to fit-model-simple, this example uses blendshapes, contour-
fitting, and can iterate the fitting.

See python demo in repo: https://github.com/patrikhuber/eos/blob/master/python/demo.py

68 ibug landmarks are loaded from the .pts file and converted
to vertex indices using the LandmarkMapper.
"""

EOS_DIR = os.path.join(os.path.dirname(__file__), 'eos')
EOS_MODEL = os.path.join(EOS_DIR, 'sfm_shape_3448.bin')
EOS_BLENDSHAPES = os.path.join(EOS_DIR, 'expression_blendshapes_3448.bin')
EOS_MAPPER = os.path.join(EOS_DIR, 'ibug_to_sfm.txt')
EOS_EDGE_TOPO = os.path.join(EOS_DIR, 'sfm_3448_edge_topology.json')
EOS_CONTOURS = os.path.join(EOS_DIR, 'sfm_model_contours.json')


def get_mesh_mask(image_shape, image_landmarks, ie_polys=None):
    """
    Gets a full-face mask from aligning a mesh to the facial landmarks
    :param image_shape:
    :param image_landmarks:
    :param ie_polys:
    :return:
    """
    mesh, pose = _predict_3d_mesh(image_landmarks, image_shape)

    projected = _project_points(mesh, pose, image_shape)
    points = _center_and_reduce_to_2d(projected, image_shape)

    return _create_mask(points, mesh.tvi, image_shape, ie_polys)


def get_mesh_landmarks(landmarks, image):
    """
    Purely for testing
    :param landmarks:
    :param image:
    :return:
    """
    mesh, pose = _predict_3d_mesh(landmarks, image.shape)

    projected = _project_points(mesh, pose, image.shape)
    points = _center_and_reduce_to_2d(projected, image.shape)

    isomap = _get_texture(mesh, pose, image)

    mask = _create_mask(points, mesh.tvi, image.shape)

    return points, isomap, mask


def _format_landmarks_for_eos(landmarks):
    """
    Formats landmarks for eos
    :param landmarks:
    :return:
    """
    eos_landmarks = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for coords in landmarks:
        eos_landmarks.append(eos.core.Landmark(str(ibug_index), [coords[0], coords[1]]))
        ibug_index = ibug_index + 1
    return eos_landmarks


def _predict_3d_mesh(landmarks, image_shape):
    """
    Predicts the 3D mesh using landmarks
    :param landmarks:
    :param image_shape:
    :return:
    """
    image_height, image_width, _ = image_shape
    model = eos.morphablemodel.load_model(EOS_MODEL)

    # The expression blendshapes:
    blendshapes = eos.morphablemodel.load_blendshapes(EOS_BLENDSHAPES)

    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    # morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
    #                                                                     color_model=eos.morphablemodel.PcaModel(),
    #                                                                     vertex_definitions=None,
    #                                                                     texture_coordinates=model.get_texture_coordinates())

    # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
    morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                        color_model=eos.morphablemodel.PcaModel(),
                                                                        vertex_definitions=None,
                                                                        texture_coordinates=model.get_texture_coordinates())

    # The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    landmark_mapper = eos.core.LandmarkMapper(EOS_MAPPER)

    # The edge topology is used to speed up computation of the occluding face contour fitting:
    edge_topology = eos.morphablemodel.load_edge_topology(EOS_EDGE_TOPO)

    # These two are used to fit the front-facing contour to the ibug contour landmarks:
    contour_landmarks = eos.fitting.ContourLandmarks.load(EOS_MAPPER)
    model_contour = eos.fitting.ModelContour.load(EOS_CONTOURS)

    # Formats the landmarks for eos
    eos_landmarks = _format_landmarks_for_eos(landmarks)

    # Fit the model, get back a mesh and the pose:
    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
                                                                                   eos_landmarks, landmark_mapper,
                                                                                   image_width, image_height,
                                                                                   edge_topology, contour_landmarks,
                                                                                   model_contour)
    # can be saved as *.obj, *.isomap.png
    return mesh, pose


def _get_pitch_yaw_roll(pose):
    pitch, yaw, roll = pose.get_rotation_euler_angles()
    return pitch, yaw, roll


# Extract the texture from the image using given mesh and camera parameters:
def _get_texture(mesh, pose, image, resolution=512):
    return eos.render.extract_texture(mesh, pose, image, isomap_resolution=resolution)


# based on https://github.com/patrikhuber/eos/issues/140#issuecomment-314775288
def _get_opencv_viewport(image_shape):
    height, width, _ = image_shape
    return np.array([0, height, width, -height])


def _get_viewport_matrix(image_shape):
    viewport = _get_opencv_viewport(image_shape)
    viewport_matrix = np.zeros((4, 4))
    viewport_matrix[0, 0] = 0.5 * viewport[2]
    viewport_matrix[3, 0] = 0.5 * viewport[2] + viewport[0]
    viewport_matrix[1, 1] = 0.5 * viewport[3]
    viewport_matrix[3, 1] = 0.5 * viewport[3] + viewport[1]
    viewport_matrix[2, 2] = 0.5
    viewport_matrix[3, 2] = 0.5
    return viewport_matrix


def _project_points(mesh, pose, image_shape):
    """
    Projects mesh points back into 2D
    :param mesh:
    :param pose:
    :param image_shape:
    :return:
    """
    # project through pose
    points = np.asarray(mesh.vertices)
    vpm = _get_viewport_matrix(image_shape)
    projection = pose.get_projection()
    modelview = pose.get_modelview()

    points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return np.asarray([vpm.dot(projection).dot(modelview).dot(point) for point in points])


def _center_and_reduce_to_2d(points, image_shape):
    """
    Centers the points on image, and reduces quaternion to 2D
    :param points:
    :param image_shape:
    :return:
    """
    height, width, _ = image_shape
    return points[:, :2] + [width / 2, height / 2]


def _create_mask(points, tvi, image_shape, ie_polys=None):
    """
    Creates a mask using the mesh vertices and their triangular face indices
    :param points: The mesh vertices, projected in 2D, shape (N, 2)
    :param tvi: the triangular vertex indices, shape (N, 3, 1)
    :param image_shape: height, width, channels of image
    :param ie_polys:
    :return: mask of points covered by mesh
    """
    mask = np.zeros((image_shape[:2] + (1,)), dtype=np.uint8)
    triangles = points[tvi]
    mouth = points[MOUTH_SFM_LANDMARKS]

    triangles = triangles[_is_triangle_ccw(triangles)]  # filter out the backfaces

    np.rint(triangles, out=triangles)
    triangles = triangles.astype(np.int32)

    np.rint(mouth, out=mouth)
    mouth = mouth.astype(np.int32)

    cv2.fillPoly(mask, triangles, (255,))
    cv2.fillPoly(mask, [mouth], (255,))

    contours, hierarchy = cv2.findContours(np.copy(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, 0, (255,), thickness=-1)

    mask = mask.astype(np.float32) / 255

    if ie_polys is not None:
        ie_polys.overlay_mask(mask)

    return mask


def _is_triangle_ccw(triangle_vertices):
    """
    Returns the boolean masks for an array of triangular vertices, testing whether a face is clockwise (facing away)
    or counter-clockwise (facing towards camera). Compares the slope of the first two segments
    :param triangle_vertices: numpy array of shape (N, 3, 2)
    :return: numpy boolean mask of shape (N, )
    """
    vert_0_x, vert_0_y = triangle_vertices[:, 0, 0], triangle_vertices[:, 0, 1]
    vert_1_x, vert_1_y = triangle_vertices[:, 1, 0], triangle_vertices[:, 1, 1]
    vert_2_x, vert_2_y = triangle_vertices[:, 2, 0], triangle_vertices[:, 2, 1]

    return ((vert_1_y - vert_0_y) * (vert_2_x - vert_1_x)) > ((vert_2_y - vert_1_y) * (vert_1_x - vert_0_x))


""" The mesh landmarks surrounding the mouth (unfilled in mesh) """
MOUTH_SFM_LANDMARKS = [
    398,
    3446,
    408,
    3253,
    406,
    3164,
    404,
    3115,
    402,
    3257,
    399,
    3374,
    442,
    3376,
    813,
    3260,
    815,
    3119,
    817,
    3167,
    819,
    3256,
    821,
    3447,
    812,
    3427,
    823,
    3332,
    826,
    3157,
    828,
    3212,
    830,
    3382,
    832,
    3391,
    423,
    3388,
    420,
    3381,
    418,
    3211,
    416,
    3155,
    414,
    3331,
    410,
    3426,
]
