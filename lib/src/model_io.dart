import 'dart:convert';
import 'dart:typed_data';

import 'model/rdst_model.dart';
import 'model/shapelet_params.dart';

/// Loads and saves [RdstModel] instances from/to JSON.
///
/// The expected JSON schema is documented in the package README and is
/// produced by `python/export_model.py`.
class ModelIo {
  ModelIo._();

  // ---------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------

  /// Parses an [RdstModel] from a JSON string.
  static RdstModel fromJson(String jsonString) {
    final Map<String, dynamic> json =
        (jsonDecode(jsonString) as Map<String, dynamic>);
    return _parseModel(json);
  }

  static RdstModel _parseModel(Map<String, dynamic> json) {
    final version = json['version'] as String;
    final nShapelets = json['nShapelets'] as int? ?? json['n_shapelets'] as int;
    final nChannels = json['nChannels'] as int? ?? json['n_channels'] as int;

    final shapletsJson = json['shapelets'] as Map<String, dynamic>;
    final shapelets = _parseShapelets(shapletsJson, nShapelets, nChannels);

    final scalerJson = json['scaler'] as Map<String, dynamic>;
    final scaler = _parseScaler(scalerJson);

    final classifierJson = json['classifier'] as Map<String, dynamic>;
    final classifier = _parseClassifier(classifierJson);

    return RdstModel(
      version: version,
      nShapelets: nShapelets,
      nChannels: nChannels,
      shapelets: shapelets,
      scaler: scaler,
      classifier: classifier,
    );
  }

  static List<ShapeletParams> _parseShapelets(
    Map<String, dynamic> json,
    int nShapelets,
    int nChannels,
  ) {
    final valuesJson = json['values'] as List<dynamic>;
    final lengths = _intList(json['lengths'] as List<dynamic>);
    final dilations = _intList(json['dilations'] as List<dynamic>);
    final thresholds = _float64List(json['thresholds'] as List<dynamic>);
    final normalise = (json['normalise'] as List<dynamic>)
        .map((v) => v as bool)
        .toList(growable: false);
    final meansJson = json['means'] as List<dynamic>;
    final stdsJson = json['stds'] as List<dynamic>;

    final shapelets = <ShapeletParams>[];
    for (var i = 0; i < nShapelets; i++) {
      final length = lengths[i];
      // valuesJson[i] is List[nChannels][length]
      final channelValues = valuesJson[i] as List<dynamic>;
      final flatValues = Float64List(nChannels * length);
      for (var c = 0; c < nChannels; c++) {
        final row = channelValues[c] as List<dynamic>;
        for (var j = 0; j < length; j++) {
          flatValues[c * length + j] = (row[j] as num).toDouble();
        }
      }

      final means = _float64List(meansJson[i] as List<dynamic>);
      final stds = _float64List(stdsJson[i] as List<dynamic>);

      shapelets.add(ShapeletParams(
        values: flatValues,
        nChannels: nChannels,
        length: length,
        dilation: dilations[i],
        threshold: thresholds[i],
        normalise: normalise[i],
        means: means,
        stds: stds,
      ));
    }
    return shapelets;
  }

  static ScalerParams _parseScaler(Map<String, dynamic> json) {
    return ScalerParams(
      mean: _float64List(json['mean'] as List<dynamic>),
      scale: _float64List(json['scale'] as List<dynamic>),
    );
  }

  static RidgeParams _parseClassifier(Map<String, dynamic> json) {
    final coefJson = json['coef'] as List<dynamic>;
    final interceptJson = json['intercept'] as List<dynamic>;
    final classes = (json['classes'] as List<dynamic>)
        .map((c) => c as String)
        .toList(growable: false);

    // coefJson is List[nRows][nCols]
    final nRows = coefJson.length;
    final nCols = (coefJson[0] as List<dynamic>).length;
    final flatCoef = Float64List(nRows * nCols);
    for (var r = 0; r < nRows; r++) {
      final row = coefJson[r] as List<dynamic>;
      for (var c = 0; c < nCols; c++) {
        flatCoef[r * nCols + c] = (row[c] as num).toDouble();
      }
    }

    return RidgeParams(
      coef: flatCoef,
      nRows: nRows,
      nCols: nCols,
      intercept: _float64List(interceptJson),
      classes: classes,
    );
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  static Float64List _float64List(List<dynamic> list) {
    final result = Float64List(list.length);
    for (var i = 0; i < list.length; i++) {
      result[i] = (list[i] as num).toDouble();
    }
    return result;
  }

  static List<int> _intList(List<dynamic> list) =>
      list.map((v) => (v as num).toInt()).toList(growable: false);
}
