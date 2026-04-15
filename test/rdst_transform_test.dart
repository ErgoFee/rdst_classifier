import 'dart:io';
import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Helper: load a fixture JSON file relative to the package root.
// ---------------------------------------------------------------------------
String _fixtureJson(String name) {
  return File('test/fixtures/$name').readAsStringSync();
}

void main() {
  group('RdstTransform', () {
    late RdstModel binaryModel;
    late RdstModel multiModel;

    setUpAll(() {
      binaryModel = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      multiModel = ModelIo.fromJson(_fixtureJson('multiclass_model.json'));
    });

    test('output shape is (nSamples, 3 * nShapelets)', () {
      const nSamples = 2;
      final nChannels = binaryModel.nChannels;
      const nTimepoints = 100;
      final X = Float64List(nSamples * nChannels * nTimepoints);

      const transform = RdstTransform();
      final result =
          transform.transform(X, nSamples, nChannels, nTimepoints, binaryModel);
      expect(result.length, equals(nSamples * 3 * binaryModel.nShapelets));
    });

    test('all-zero input produces finite features', () {
      const nSamples = 1;
      final nChannels = binaryModel.nChannels;
      const nTimepoints = 100;
      final X = Float64List(nSamples * nChannels * nTimepoints);

      const transform = RdstTransform();
      final result =
          transform.transform(X, nSamples, nChannels, nTimepoints, binaryModel);
      for (final v in result) {
        expect(v.isFinite, isTrue, reason: 'Expected finite feature, got $v');
      }
    });

    test('multiclass model — output shape is (nSamples, 3*nShapelets)', () {
      const nSamples = 3;
      final nChannels = multiModel.nChannels;
      const nTimepoints = 120;
      final X = Float64List(nSamples * nChannels * nTimepoints);

      const transform = RdstTransform();
      final result =
          transform.transform(X, nSamples, nChannels, nTimepoints, multiModel);
      expect(result.length, equals(nSamples * 3 * multiModel.nShapelets));
    });

    test('different samples produce different features', () {
      final nChannels = binaryModel.nChannels;
      const nTimepoints = 100;
      // sample0: zeros; sample1: ramp 0..99
      final X = Float64List(2 * nChannels * nTimepoints);
      for (var c = 0; c < nChannels; c++) {
        for (var t = 0; t < nTimepoints; t++) {
          X[1 * nChannels * nTimepoints + c * nTimepoints + t] = t.toDouble();
        }
      }

      const transform = RdstTransform();
      final result =
          transform.transform(X, 2, nChannels, nTimepoints, binaryModel);
      final nFeatures = 3 * binaryModel.nShapelets;

      // At least one feature should differ between the two samples.
      var anyDifferent = false;
      for (var j = 0; j < nFeatures; j++) {
        if ((result[j] - result[nFeatures + j]).abs() > 1e-12) {
          anyDifferent = true;
          break;
        }
      }
      expect(anyDifferent, isTrue);
    });
  });
}
