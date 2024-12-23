import taichi as ti

ti.init(arch=ti.cuda)


@ti.kernel
def merge_sort_split(
    key1: ti.types.ndarray(ti.float32, ndim=1), key2: ti.types.ndarray(ti.float32, ndim=1)
) -> ti.types.ndarray(ti.float32, ndim=1):
    """
    this function is used to merge two sorted array

    key1: sorted array
    key2: sorted array

    return: sorted array
    """

    pass
