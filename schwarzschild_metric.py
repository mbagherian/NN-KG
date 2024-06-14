import tensorflow as tf

def schwarzschild_metric(r, M):
  """Calculates the Schwarzschild metric components. 

  Args:
    r: Radial coordinate.
    M: Mass of the black hole.

  Returns:
    g_tt: Time-time component of the metric.
    g_rr: Radial-radial component of the metric.
  """

    g_tt = -(1 - 2 * tf.constant(G, dtype=tf.float32) * M / (tf.constant(c, dtype=tf.float32)**2 * r))
    g_rr = 1 / (1 - 2 * tf.constant(G, dtype=tf.float32) * M / (tf.constant(c, dtype=tf.float32)**2 * r))

    return g_tt, g_rr
