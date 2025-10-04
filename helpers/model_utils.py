import tensorflow as tf
import tensorflow_hub as hub
import os
import time

# Global cache for model to avoid reloading
_model_cache = None

def load_movenet_model(custom_model_path=None):
    """Loads MoveNet SinglePose Lightning SavedModel with minimal optimizations."""
    global _model_cache
    
    # Return cached model if available
    if _model_cache is not None:
        print("Using cached MoveNet model (instant loading)...")
        return _model_cache
    
    start_time = time.time()
    
    print("Loading MoveNet Lightning model...")
    # Official TF Hub SavedModel for MoveNet SinglePose Lightning
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

    # Resolve to local cache path (downloads once, then reused offline)
    print("Resolving model path...")
    resolve_start = time.time()
    resolved_path = hub.resolve(model_url)
    resolve_time = time.time() - resolve_start
    print(f"Model path resolved in {resolve_time:.2f} seconds")
    
    print("Loading SavedModel...")
    load_start = time.time()
    module = tf.saved_model.load(resolved_path)
    load_time = time.time() - load_start
    print(f"SavedModel loaded in {load_time:.2f} seconds")
    
    serving_fn = module.signatures["serving_default"]

    input_size = 192  # Lightning expects 192x192

    def movenet(input_image: tf.Tensor):
        # MoveNet SavedModel expects int32 input in range [0,255] with shape [1, 192, 192, 3]
        input_image_int = tf.cast(input_image, dtype=tf.int32)
        outputs = serving_fn(input_image_int)
        # Output key is typically 'output_0'
        keypoints_with_scores = outputs["output_0"].numpy()
        return keypoints_with_scores

    # Cache the result
    _model_cache = (movenet, input_size)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return movenet, input_size
