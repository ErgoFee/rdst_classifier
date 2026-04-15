import 'dart:typed_data';

import 'model/rdst_model.dart';
import 'utils/math_utils.dart';

/// Pure-Dart port of the Ridge linear classifier inference step.
///
/// Supports both binary and multi-class classification.
///
/// Binary:
///   - Decision scores: `s = X @ coef[0] + intercept[0]`  (1-D)
///   - Prediction: `s > 0 → classes[1]`, else `classes[0]`
///   - Probabilities: sigmoid → `[1-p, p]`
///
/// Multi-class:
///   - Decision scores: `S = X @ coef.T + intercept`  shape `(nSamples, nClasses)`
///   - Prediction: `argmax(S[i])`
///   - Probabilities: softmax(S[i])
class RidgeClassifier {
  const RidgeClassifier();

  // ---------------------------------------------------------------------------
  // Prediction
  // ---------------------------------------------------------------------------

  /// Returns a hard label for each sample in [X].
  ///
  /// [X] has shape `(nSamples, nFeatures)` stored row-major.
  List<String> predict(
    Float64List X,
    int nSamples,
    int nFeatures,
    RidgeParams params,
  ) {
    final predictions = <String>[];
    if (params.isBinary) {
      for (var i = 0; i < nSamples; i++) {
        final score = _dotRow(X, i, nFeatures, params.coef, 0, params.nCols) +
            params.intercept[0];
        predictions.add(score > 0 ? params.classes[1] : params.classes[0]);
      }
    } else {
      for (var i = 0; i < nSamples; i++) {
        var bestScore = double.negativeInfinity;
        var bestClass = 0;
        for (var r = 0; r < params.nRows; r++) {
          final score = _dotRow(X, i, nFeatures, params.coef, r, params.nCols) +
              params.intercept[r];
          if (score > bestScore) {
            bestScore = score;
            bestClass = r;
          }
        }
        predictions.add(params.classes[bestClass]);
      }
    }
    return predictions;
  }

  // ---------------------------------------------------------------------------
  // Probabilities
  // ---------------------------------------------------------------------------

  /// Returns a probability vector for each sample in [X].
  ///
  /// For binary classification: `[P(classes[0]), P(classes[1])]` via sigmoid.
  /// For multi-class: softmax of linear scores.
  ///
  /// Returns a [Float64List] of shape `(nSamples, nClasses)` row-major.
  Float64List predictProba(
    Float64List X,
    int nSamples,
    int nFeatures,
    RidgeParams params,
  ) {
    final nClasses = params.nClasses;
    final result = Float64List(nSamples * nClasses);

    if (params.isBinary) {
      for (var i = 0; i < nSamples; i++) {
        final score = _dotRow(X, i, nFeatures, params.coef, 0, params.nCols) +
            params.intercept[0];
        final p = sigmoid(score);
        result[i * nClasses] = 1.0 - p; // P(classes[0])
        result[i * nClasses + 1] = p; // P(classes[1])
      }
    } else {
      final scores = Float64List(params.nRows);
      for (var i = 0; i < nSamples; i++) {
        for (var r = 0; r < params.nRows; r++) {
          scores[r] = _dotRow(X, i, nFeatures, params.coef, r, params.nCols) +
              params.intercept[r];
        }
        final proba = softmax(scores);
        for (var c = 0; c < nClasses; c++) {
          result[i * nClasses + c] = proba[c];
        }
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Helper
  // ---------------------------------------------------------------------------

  /// Dot product of row [iRow] of [X] (shape ·×[nColsX]) with row [iRow2] of
  /// [coef] (shape ·×[nColsCoef]).
  double _dotRow(
    Float64List X,
    int iRow,
    int nColsX,
    Float64List coef,
    int iRow2,
    int nColsCoef,
  ) {
    var sum = 0.0;
    for (var j = 0; j < nColsX; j++) {
      sum += X[iRow * nColsX + j] * coef[iRow2 * nColsCoef + j];
    }
    return sum;
  }
}
