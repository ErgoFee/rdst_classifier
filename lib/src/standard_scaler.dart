import 'dart:typed_data';

import 'model/rdst_model.dart';

/// Applies a fitted StandardScaler to a feature matrix.
///
/// The transform is: `(x - mean) / scale` element-wise.
class StandardScaler {
  const StandardScaler();

  /// Scales [X] in-place and returns it for convenience.
  ///
  /// [X] has shape `(nSamples, nFeatures)` stored row-major.
  /// [params] must have `mean` and `scale` arrays of length `nFeatures`.
  Float64List transform(
    Float64List X,
    int nSamples,
    int nFeatures,
    ScalerParams params,
  ) {
    final result = Float64List(X.length);
    for (var i = 0; i < nSamples; i++) {
      for (var j = 0; j < nFeatures; j++) {
        final scale = params.scale[j];
        final v = X[i * nFeatures + j];
        result[i * nFeatures + j] =
            scale == 0.0 ? 0.0 : (v - params.mean[j]) / scale;
      }
    }
    return result;
  }
}
