import numpy as np
import scipy.special as sps
from warnings import warn

def cart2sphere(x, y, z):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    Parameters
    ------------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    ---------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle
    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi



def real_sph_harm(m, n, theta, phi):
    """ Compute real spherical harmonics.

    Where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)     if m > 0
        $Y^0_n$                     if m = 0
        Real($Y^|m|_n$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    See Also
    --------
    scipy.special.sph_harm
    """
    return real_sh_descoteaux_from_index(m, n, theta, phi, legacy=True)
 


def real_sh_descoteaux_from_index(m, n, theta, phi, legacy=True):
    """ Compute real spherical harmonics as in Descoteaux et al. 2007 [1]_,
    where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)      if m > 0
        $Y^0_n$                      if m = 0
        Real($Y^m_n$) * sqrt(2)      if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    legacy: bool, optional
        If true, uses DIPY's legacy descoteaux07 implementation (where |m|
        is used for m < 0). Else, implements the basis as defined in
        Descoteaux et al. 2007 (without the absolute value).

    Returns
    -------
    real_sh : real float
        The real harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    References
    ----------
     .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
           Regularized, Fast, and Robust Analytical Q-ball Imaging.
           Magn. Reson. Med. 2007;58:497-510.
    """
    if legacy:
        # In the case where m < 0, legacy descoteaux basis considers |m|
        warn('The legacy descoteaux07 SH basis is outdated and will be '
             'deprecated in a future DIPY release. Consider using the new '
             'descoteaux07 basis.', category=PendingDeprecationWarning)
        sh = spherical_harmonics(np.abs(m), n, phi, theta)
    else:
        # In the cited paper, the basis is defined without the absolute value
        sh = spherical_harmonics(m, n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))

    return real_sh
 

def spherical_harmonics(m, n, theta, phi, use_scipy=True):
    """Compute spherical harmonics.

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.
    use_scipy : bool, optional
        If True, use scipy implementation.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function.

    The usual definitions for ``theta` and `phi`` used in DIPY are interchanged
    in the method definition to agree with the definitions in
    scipy.special.sph_harm, where `theta` represents the azimuthal coordinate
    and `phi` represents the polar coordinate.

    Altough scipy uses a naming convention where ``m`` is the order and ``n``
    is the degree of the SH, the opposite of DIPY's, their definition for both
    parameters is the same as ours, with ``n >= 0`` and ``|m| <= n``.
    """
    if use_scipy:
        return sps.sph_harm(m, n, theta, phi, dtype=complex)

    x = np.cos(phi)
    val = sps.lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (sps.gammaln(n - m + 1) - sps.gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val