# Object Detection API compatibility shims for newer TensorFlow
import tensorflow.compat.v2 as tf

try:
    # TF-Slim uses internal TF APIs: tensorflow.python.ops.control_flow_ops.case/cond
    from tensorflow.python.ops import control_flow_ops  # internal

    if not hasattr(control_flow_ops, "case"):
        control_flow_ops.case = tf.case

    if not hasattr(control_flow_ops, "cond"):
        control_flow_ops.cond = tf.cond

except Exception:
    # Keep import-safe; training will fail later if symbols still missing.
    pass
