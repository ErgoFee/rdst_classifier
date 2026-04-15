import 'dart:typed_data';

import 'shapelet_params.dart';

/// Parameters for the StandardScaler step.
///
/// Applies: `(x - mean) / scale` element-wise.
class ScalerParams {
  /// Per-feature mean.  Length = `3 * nShapelets`.
  final Float64List mean;

  /// Per-feature scale.  Length = `3 * nShapelets`.
  final Float64List scale;

  const ScalerParams({required this.mean, required this.scale});
}

/// Parameters for the Ridge linear classifier.
///
/// For **binary** classification `coef` has shape `(1, nFeatures)` and
/// `intercept` has length `1`. Decision rule: score > 0 → `classes[1]`,
/// else `classes[0]`.
///
/// For **multi-class** `coef` has shape `(nClasses, nFeatures)` and
/// `intercept` has length `nClasses`. Decision rule: argmax of scores.
class RidgeParams {
  /// Coefficient matrix stored **row-major** (classIndex × feature).
  /// Flat length = `nClassesOrOne * nFeatures`.
  final Float64List coef;

  /// Number of rows in [coef] (1 for binary, nClasses for multi-class).
  final int nRows;

  /// Number of columns in [coef] (= nFeatures = 3 * nShapelets).
  final int nCols;

  /// Bias / intercept.  Length = [nRows].
  final Float64List intercept;

  /// Ordered class labels, length = nClasses (2+ for binary/multi-class).
  final List<String> classes;

  const RidgeParams({
    required this.coef,
    required this.nRows,
    required this.nCols,
    required this.intercept,
    required this.classes,
  });

  /// Convenience: number of distinct output classes.
  int get nClasses => classes.length;

  /// Whether this is a binary classification problem.
  bool get isBinary => classes.length == 2;

  /// Returns coef[row][col].
  double coefAt(int row, int col) => coef[row * nCols + col];
}

/// Complete RDST model, combining all fitted parameters.
class RdstModel {
  /// Format version string from the JSON file.
  final String version;

  /// Total number of shapelets.
  final int nShapelets;

  /// Number of input channels (time-series dimensions).
  final int nChannels;

  /// Shapelet parameters, one per shapelet.
  final List<ShapeletParams> shapelets;

  /// Fitted StandardScaler parameters.
  final ScalerParams scaler;

  /// Fitted Ridge classifier parameters.
  final RidgeParams classifier;

  const RdstModel({
    required this.version,
    required this.nShapelets,
    required this.nChannels,
    required this.shapelets,
    required this.scaler,
    required this.classifier,
  });
}
