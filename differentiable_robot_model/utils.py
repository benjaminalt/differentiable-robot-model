# Copyright (c) Facebook, Inc. and its affiliates.
"""
Utils
====================================
TODO
"""

import operator
from functools import reduce

import numpy as np
import torch
from .external.pytorch3d_transformations import matrix_to_euler_angles, quaternion_to_matrix

prod = lambda l: reduce(operator.mul, l, 1)


def cross_product(vec3a, vec3b):
    vec3a = convert_into_at_least_2d_pytorch_tensor(vec3a)
    vec3b = convert_into_at_least_2d_pytorch_tensor(vec3b)
    skew_symm_mat_a = vector3_to_skew_symm_matrix(vec3a)
    return (skew_symm_mat_a @ vec3b.unsqueeze(2)).squeeze(2)


def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def vector3_to_skew_symm_matrix(vec3):
    vec3 = convert_into_at_least_2d_pytorch_tensor(vec3)
    batch_size = vec3.shape[0]
    skew_symm_mat = vec3.new_zeros((batch_size, 3, 3))
    skew_symm_mat[:, 0, 1] = -vec3[:, 2]
    skew_symm_mat[:, 0, 2] = vec3[:, 1]
    skew_symm_mat[:, 1, 0] = vec3[:, 2]
    skew_symm_mat[:, 1, 2] = -vec3[:, 0]
    skew_symm_mat[:, 2, 0] = -vec3[:, 1]
    skew_symm_mat[:, 2, 1] = vec3[:, 0]
    return skew_symm_mat


def torch_square(x):
    return x * x


def exp_map_so3(omega, epsilon=1.0e-14):
    omegahat = vector3_to_skew_symm_matrix(omega).squeeze()

    norm_omega = torch.norm(omega, p=2)
    exp_omegahat = (
        torch.eye(3)
        + ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat)
        + (((1.0 - torch.cos(norm_omega)) / (torch_square(norm_omega + epsilon))) * (omegahat @ omegahat))
    )
    return exp_omegahat


def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable)
    else:
        return torch.Tensor(variable)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var


def sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(quaternions)


def pose_to_affine(pose: torch.Tensor) -> torch.Tensor:
    assert pose.shape[-1] == 7
    rot = quaternion_to_rotation_matrix(pose[..., 3:])
    trans = pose[..., :3].unsqueeze(-1)
    # torch.eye doesn't work as it does not retain grads
    affine = torch.as_tensor([0, 0, 0, 1], dtype=pose.dtype, device=pose.device).repeat((*pose.shape[:-1], 1, 1))
    affine = torch.concat((torch.concat((rot, trans), dim=-1), affine), dim=-2)
    return affine


def quaternion_to_euler_zyx(q: torch.Tensor) -> torch.Tensor:
    """
    Convert q quaternion (w,x,y,z) to Euler angles (ZYX convention, in radians, normalized to [-pi, pi])
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    :param q: Quaternion (w,x,y,z)
    :return Euler angles (ZYX convention, in radians, normalized to [-pi, pi])
    """
    rotations = matrix_to_euler_angles(quaternion_to_matrix(q), convention="ZYX").flip(-1)
    return rotations