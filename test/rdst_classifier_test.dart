import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Helper: load a fixture JSON file relative to the package root.
// ---------------------------------------------------------------------------
String _fixtureJson(String name) {
  // When run via `dart test` the working directory is the package root.
  return File('test/fixtures/$name').readAsStringSync();
}

/// Converts a nested List<dynamic> (3-D: samples × channels × timepoints)
/// to a flat [Float64List] in sample-major, channel-major order.
Float64List _toFlat(List<dynamic> raw) {
  final nSamples = raw.length;
  final nChannels = (raw[0] as List<dynamic>).length;
  final nTimepoints = ((raw[0] as List<dynamic>)[0] as List<dynamic>).length;
  final result = Float64List(nSamples * nChannels * nTimepoints);
  for (var s = 0; s < nSamples; s++) {
    final sample = raw[s] as List<dynamic>;
    for (var c = 0; c < nChannels; c++) {
      final ch = sample[c] as List<dynamic>;
      for (var t = 0; t < nTimepoints; t++) {
        result[s * nChannels * nTimepoints + c * nTimepoints + t] =
            (ch[t] as num).toDouble();
      }
    }
  }
  return result;
}

void main() {
  late Map<String, dynamic> fixtures;

  setUpAll(() {
    fixtures = jsonDecode(_fixtureJson('expected_predictions.json'))
        as Map<String, dynamic>;
  });

  // ---------------------------------------------------------------------------
  // Binary model end-to-end
  // ---------------------------------------------------------------------------
  group('RdstClassifier — binary end-to-end', () {
    late RdstClassifier clf;
    late Map<String, dynamic> binaryFixture;

    setUpAll(() {
      clf = RdstClassifier.fromJson(_fixtureJson('binary_model.json'));
      binaryFixture = fixtures['binary'] as Map<String, dynamic>;
    });

    test('classes list matches fixture', () {
      final expected =
          (binaryFixture['classes'] as List<dynamic>).cast<String>();
      expect(clf.classes, equals(expected));
    });

    test('predict matches Python output', () {
      final rawX = binaryFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final predictions = clf.predict(X, nSamples, nChannels, nTimepoints);
      final expected = (binaryFixture['expected_predictions'] as List<dynamic>)
          .cast<String>();

      expect(predictions, equals(expected));
    });

    test('predictProba has correct shape', () {
      final rawX = binaryFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      expect(probas.length, equals(nSamples * 2));
    });

    test('predictProba rows sum to 1', () {
      final rawX = binaryFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      for (var i = 0; i < nSamples; i++) {
        expect(probas[i * 2] + probas[i * 2 + 1], closeTo(1.0, 1e-6));
      }
    });

    test('predictProba matches Python probabilities within tolerance', () {
      final rawX = binaryFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      final expectedProbas = binaryFixture['expected_probas'] as List<dynamic>;

      for (var i = 0; i < nSamples; i++) {
        final expRow = expectedProbas[i] as List<dynamic>;
        for (var c = 0; c < 2; c++) {
          expect(
            probas[i * 2 + c],
            closeTo((expRow[c] as num).toDouble(), 1e-4),
            reason: 'sample=$i class=$c',
          );
        }
      }
    });
  });

  // ---------------------------------------------------------------------------
  // Multiclass model end-to-end
  // ---------------------------------------------------------------------------
  group('RdstClassifier — multiclass end-to-end', () {
    late RdstClassifier clf;
    late Map<String, dynamic> multiFixture;

    setUpAll(() {
      clf = RdstClassifier.fromJson(_fixtureJson('multiclass_model.json'));
      multiFixture = fixtures['multiclass'] as Map<String, dynamic>;
    });

    test('classes list matches fixture', () {
      final expected =
          (multiFixture['classes'] as List<dynamic>).cast<String>();
      expect(clf.classes, equals(expected));
    });

    test('predict matches Python output', () {
      final rawX = multiFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final predictions = clf.predict(X, nSamples, nChannels, nTimepoints);
      final expected = (multiFixture['expected_predictions'] as List<dynamic>)
          .cast<String>();

      expect(predictions, equals(expected));
    });

    test('predictProba has correct shape', () {
      final rawX = multiFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      final nClasses = clf.classes.length;
      expect(probas.length, equals(nSamples * nClasses));
    });

    test('predictProba rows sum to 1', () {
      final rawX = multiFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final nClasses = clf.classes.length;
      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      for (var i = 0; i < nSamples; i++) {
        double rowSum = 0;
        for (var c = 0; c < nClasses; c++) {
          rowSum += probas[i * nClasses + c];
        }
        expect(rowSum, closeTo(1.0, 1e-6));
      }
    });

    test('predictProba matches Python probabilities within tolerance', () {
      final rawX = multiFixture['test_X'] as List<dynamic>;
      final nSamples = rawX.length;
      final nChannels = (rawX[0] as List<dynamic>).length;
      final nTimepoints =
          ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
      final X = _toFlat(rawX);

      final nClasses = clf.classes.length;
      final probas = clf.predictProba(X, nSamples, nChannels, nTimepoints);
      final expectedProbas = multiFixture['expected_probas'] as List<dynamic>;

      for (var i = 0; i < nSamples; i++) {
        final expRow = expectedProbas[i] as List<dynamic>;
        for (var c = 0; c < nClasses; c++) {
          expect(
            probas[i * nClasses + c],
            closeTo((expRow[c] as num).toDouble(), 1e-4),
            reason: 'sample=$i class=$c',
          );
        }
      }
    });
  });
}
