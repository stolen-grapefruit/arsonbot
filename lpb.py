import numpy as np

def lpb(t0, tf, q0, qf, tb=None, qdd=None, N=100):
    time = tf - t0
    t = np.linspace(t0, tf, N)

    if tb == 0 and qdd is not None:
        under_sqrt = qdd**2 * time**2 - 4 * abs(qdd) * abs(qf - q0)
        if under_sqrt < 0:
            raise ValueError("Insufficient acceleration available. Cannot compute blend time.")
        tb = (time / 2) - (np.sqrt(under_sqrt) / (2 * qdd))

    elif tb is not None and (qdd is None or qdd == 0):
        qdd = abs((qf - q0) / (tb * (time - tb)))

    if tb is None or qdd is None:
        raise ValueError("Either tb or qdd must be provided and non-zero.")

    # Determine blend coefficients
    sign = 1 if q0 < qf else -1
    ab0_rhs = np.array([q0, 0, sign * qdd])
    abf_rhs = np.array([qf, 0, -sign * qdd])

    A = np.array([
        [1, t0, t0**2],
        [0, 1, 2*t0],
        [0, 0, 2]
    ])
    ab0 = np.linalg.solve(A, ab0_rhs)
    abf = np.linalg.solve(A, abf_rhs)

    # Blending boundary values
    qb1 = ab0[0] + ab0[1] * (t0 + tb) + ab0[2] * (t0 + tb)**2
    qb2 = abf[0] + abf[1] * (tf - tb) + abf[2] * (tf - tb)**2

    a = np.linalg.solve(
        np.array([[1, t0 + tb], [1, tf - tb]]),
        np.array([qb1, qb2])
    )

    # Segment indices
    t11 = t[(t0 <= t) & (t <= t0 + tb)]
    t22 = t[(t0 + tb < t) & (t < tf - tb)]
    t33 = t[(tf - tb <= t) & (t <= tf)]

    # First parabolic
    q1 = ab0[0] + ab0[1] * t11 + ab0[2] * t11**2
    qd1 = ab0[1] + 2 * ab0[2] * t11
    qdd1 = np.full_like(t11, 2 * ab0[2])

    # Linear segment
    q2 = a[0] + a[1] * t22
    qd2 = np.full_like(t22, a[1])
    qdd2 = np.zeros_like(t22)

    # Second parabolic
    q3 = abf[0] + abf[1] * t33 + abf[2] * t33**2
    qd3 = abf[1] + 2 * abf[2] * t33
    qdd3 = np.full_like(t33, 2 * abf[2])

    # Concatenate segments
    q = np.concatenate([q1, q2, q3])
    qd = np.concatenate([qd1, qd2, qd3])
    qdd_out = np.concatenate([qdd1, qdd2, qdd3])
    qddd = np.zeros(N)

    return t, q, qd, qdd_out, qddd


# Example usage:
if __name__ == "__main__":
    t, q, qd, qdd, qddd = lpb(t0=0, tf=5, q0=0, qf=1, tb=1, qdd=None, N=200)

    import matplotlib.pyplot as plt
    plt.plot(t, q, label='q')
    plt.plot(t, qd, label='qd')
    plt.plot(t, qdd, label='qdd')
    plt.title("Linear Parabolic Blend")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.show()
