import 'dart:typed_data';

import 'model/rdst_model.dart';
import 'model_io.dart';
import 'rdst_transform.dart';
import 'ridge_classifier.dart';
import 'standard_scaler.dart';

/// High-level inference pipeline for the Random Dilated Shapelet Transform
/// (RDST) classifier.
///
/// Chains together:
///   1. [RdstTransform]   — dilated shapelet feature extraction
///   2. [StandardScaler]  — zero-mean unit-variance scaling
///   3. [RidgeClassifier] — linear classifier with optional probability output
///
/// Usage:
/// ```dart
/// final classifier = RdstClassifier.fromJson(jsonString);
/// final labels = classifier.predict(X, nSamples, nChannels, nTimepoints);
/// final probas = classifier.predictProba(X, nSamples, nChannels, nTimepoints);
/// ```
class RdstClassifier {
  /// The loaded model parameters.
  final RdstModel model;

  final RdstTransform _transform;
  final StandardScaler _scaler;
  final RidgeClassifier _ridge;

  const RdstClassifier(
    this.model, {
    RdstTransform transform = const RdstTransform(),
    StandardScaler scaler = const StandardScaler(),
    RidgeClassifier ridge = const RidgeClassifier(),
  })  : _transform = transform,
        _scaler = scaler,
        _ridge = ridge;

  // ---------------------------------------------------------------------------
  // Factory constructors
  // ---------------------------------------------------------------------------

  /// Deserialises an [RdstClassifier] from a JSON string previously produced
  /// by the Python `export_model.py` script.
  factory RdstClassifier.fromJson(String jsonString) {
    final model = ModelIo.fromJson(jsonString);
    return RdstClassifier(model);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /// Returns the ordered list of class labels this model was trained on.
  List<String> get classes => model.classifier.classes;

  /// Predicts a hard class label for each sample in [X].
  ///
  /// [X] is a flat [Float64List] with shape `(nSamples, nChannels, nTimepoints)`
  /// in sample-major, channel-major order:
  ///   `X[s, c, t] = X[s * nChannels * nTimepoints + c * nTimepoints + t]`
  ///
  /// Returns a [List<String>] of length [nSamples].
  List<String> predict(
    Float64List X,
    int nSamples,
    int nChannels,
    int nTimepoints,
  ) {
    final features = _extractFeatures(X, nSamples, nChannels, nTimepoints);
    return _ridge.predict(
        features, nSamples, model.nShapelets * 3, model.classifier);
  }

  /// Predicts class probabilities for each sample in [X].
  ///
  /// [X] has the same layout as described in [predict].
  ///
  /// Returns a [Float64List] of shape `(nSamples, nClasses)` in row-major
  /// order, where `nClasses == classes.length`.
  ///
  /// For binary classification probabilities are derived from the sigmoid of
  /// the decision score; for multi-class from softmax.
  Float64List predictProba(
    Float64List X,
    int nSamples,
    int nChannels,
    int nTimepoints,
  ) {
    final features = _extractFeatures(X, nSamples, nChannels, nTimepoints);
    return _ridge.predictProba(
        features, nSamples, model.nShapelets * 3, model.classifier);
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /// Runs the transform + scaler pipeline and returns the scaled feature matrix
  /// of shape `(nSamples, 3 * nShapelets)`.
  Float64List _extractFeatures(
    Float64List X,
    int nSamples,
    int nChannels,
    int nTimepoints,
  ) {
    final nFeatures = model.nShapelets * 3;

    // Step 1: shapelet feature extraction.
    final raw =
        _transform.transform(X, nSamples, nChannels, nTimepoints, model);

    // Step 2: StandardScaler.
    return _scaler.transform(raw, nSamples, nFeatures, model.scaler);
  }
}
