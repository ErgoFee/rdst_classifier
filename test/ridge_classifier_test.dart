import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

RidgeParams _binaryParams({
  required List<double> coef,
  required double intercept,
  List<String> classes = const ['neg', 'pos'],
}) {
  return RidgeParams(
    coef: Float64List.fromList(coef),
    nRows: 1,
    nCols: coef.length,
    intercept: Float64List.fromList([intercept]),
    classes: classes,
  );
}

RidgeParams _multiParams({
  required List<List<double>> coef,
  required List<double> intercept,
  required List<String> classes,
}) {
  final flat = Float64List.fromList(coef.expand((row) => row).toList());
  return RidgeParams(
    coef: flat,
    nRows: coef.length,
    nCols: coef[0].length,
    intercept: Float64List.fromList(intercept),
    classes: classes,
  );
}

void main() {
  const ridge = RidgeClassifier();

  // ---------------------------------------------------------------------------
  // Binary predict
  // ---------------------------------------------------------------------------
  group('RidgeClassifier.predict — binary', () {
    test('positive score → classes[1]', () {
      // coef=[1], intercept=0: score = x[0]*1 > 0 when x[0]>0
      final params = _binaryParams(coef: [1.0], intercept: 0.0);
      final X = Float64List.fromList([2.0]); // score=2 > 0
      expect(ridge.predict(X, 1, 1, params), equals(['pos']));
    });

    test('negative score → classes[0]', () {
      final params = _binaryParams(coef: [1.0], intercept: 0.0);
      final X = Float64List.fromList([-1.0]);
      expect(ridge.predict(X, 1, 1, params), equals(['neg']));
    });

    test('intercept shifts decision boundary', () {
      // coef=[1], intercept=-5: score = x - 5
      final params = _binaryParams(coef: [1.0], intercept: -5.0);
      final X = Float64List.fromList([3.0, 7.0]); // scores: -2, +2
      final preds = ridge.predict(X, 2, 1, params);
      expect(preds[0], equals('neg'));
      expect(preds[1], equals('pos'));
    });

    test('multi-feature binary', () {
      // coef=[1,1], intercept=0: score = x0+x1
      final params = _binaryParams(coef: [1.0, 1.0], intercept: 0.0);
      final X = Float64List.fromList([0.5, 0.3, -0.5, -0.3]);
      // sample0: 0.8 > 0 → pos; sample1: -0.8 ≤ 0 → neg
      final preds = ridge.predict(X, 2, 2, params);
      expect(preds[0], equals('pos'));
      expect(preds[1], equals('neg'));
    });
  });

  // ---------------------------------------------------------------------------
  // Multi-class predict
  // ---------------------------------------------------------------------------
  group('RidgeClassifier.predict — multi-class', () {
    test('argmax class selection', () {
      final params = _multiParams(
        coef: [
          [1.0, 0.0],
          [0.0, 1.0],
          [-1.0, -1.0],
        ],
        intercept: [0.0, 0.0, 0.0],
        classes: ['A', 'B', 'C'],
      );
      // X=[2,1]: scores=[2,1,-3] → A
      final X = Float64List.fromList([2.0, 1.0]);
      expect(ridge.predict(X, 1, 2, params), equals(['A']));
    });

    test('intercept breaks ties', () {
      final params = _multiParams(
        coef: [
          [1.0],
          [1.0],
          [1.0],
        ],
        intercept: [0.0, 10.0, 0.0], // B gets +10
        classes: ['A', 'B', 'C'],
      );
      final X = Float64List.fromList([1.0]);
      expect(ridge.predict(X, 1, 1, params), equals(['B']));
    });
  });

  // ---------------------------------------------------------------------------
  // Binary predictProba
  // ---------------------------------------------------------------------------
  group('RidgeClassifier.predictProba — binary', () {
    test('probabilities sum to 1', () {
      final params = _binaryParams(coef: [2.0], intercept: 0.5);
      final X = Float64List.fromList([1.0]);
      final proba = ridge.predictProba(X, 1, 1, params);
      expect(proba[0] + proba[1], closeTo(1.0, 1e-10));
    });

    test('positive score → P(pos) > 0.5', () {
      final params = _binaryParams(coef: [1.0], intercept: 0.0);
      final X = Float64List.fromList([5.0]);
      final proba = ridge.predictProba(X, 1, 1, params);
      expect(proba[1], greaterThan(0.5));
    });

    test('zero score → P(pos) == 0.5', () {
      final params = _binaryParams(coef: [1.0], intercept: 0.0);
      final X = Float64List.fromList([0.0]);
      final proba = ridge.predictProba(X, 1, 1, params);
      expect(proba[0], closeTo(0.5, 1e-10));
      expect(proba[1], closeTo(0.5, 1e-10));
    });

    test('output shape is (nSamples, 2)', () {
      final params = _binaryParams(coef: [1.0], intercept: 0.0);
      final X = Float64List.fromList([1.0, -1.0, 0.0]);
      final proba = ridge.predictProba(X, 3, 1, params);
      expect(proba.length, equals(6));
    });
  });

  // ---------------------------------------------------------------------------
  // Multi-class predictProba
  // ---------------------------------------------------------------------------
  group('RidgeClassifier.predictProba — multi-class', () {
    test('probabilities sum to 1', () {
      final params = _multiParams(
        coef: [
          [1.0, 0.0],
          [0.0, 1.0],
          [-1.0, -1.0],
        ],
        intercept: [0.0, 0.0, 0.0],
        classes: ['A', 'B', 'C'],
      );
      final X = Float64List.fromList([2.0, 1.0]);
      final proba = ridge.predictProba(X, 1, 2, params);
      expect(proba[0] + proba[1] + proba[2], closeTo(1.0, 1e-10));
    });

    test('highest scoring class gets highest probability', () {
      final params = _multiParams(
        coef: [
          [1.0],
          [0.0],
          [-1.0],
        ],
        intercept: [0.0, 0.0, 0.0],
        classes: ['A', 'B', 'C'],
      );
      final X = Float64List.fromList([5.0]);
      final proba = ridge.predictProba(X, 1, 1, params);
      // A has score=5, B=0, C=-5 → A should dominate
      expect(proba[0], greaterThan(proba[1]));
      expect(proba[1], greaterThan(proba[2]));
    });
  });
}
