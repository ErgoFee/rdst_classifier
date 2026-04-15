/// Pure-Dart inference engine for the Random Dilated Shapelet Transform (RDST)
/// classifier.
///
/// Load a pre-trained model (exported from Python/aeon via `export_model.py`)
/// and run inference inside any Dart or Flutter application.
///
/// ## Quick start
///
/// ```dart
/// import 'dart:typed_data';
/// import 'package:rdst_classifier/rdst_classifier.dart';
///
/// // Load a model exported by the Python tooling.
/// final classifier = RdstClassifier.fromJson(jsonString);
///
/// // X: Float64List of shape (nSamples, nChannels, nTimepoints), row-major.
/// final labels = classifier.predict(X, nSamples, nChannels, nTimepoints);
/// final probas = classifier.predictProba(X, nSamples, nChannels, nTimepoints);
/// ```
library rdst_classifier;

export 'src/model/rdst_model.dart';
export 'src/model/shapelet_params.dart';
export 'src/model_io.dart';
export 'src/rdst_classifier.dart';
export 'src/rdst_transform.dart';
export 'src/ridge_classifier.dart';
export 'src/standard_scaler.dart';
export 'src/utils/math_utils.dart';
export 'src/utils/subsequence.dart';
