import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

void main() {
  group('StandardScaler.transform', () {
    const scaler = StandardScaler();

    test('zero mean and unit scale is identity', () {
      final params = ScalerParams(
        mean: Float64List.fromList([0.0, 0.0, 0.0]),
        scale: Float64List.fromList([1.0, 1.0, 1.0]),
      );
      final X = Float64List.fromList([1.0, 2.0, 3.0]);
      final result = scaler.transform(X, 1, 3, params);
      expect(result, equals([1.0, 2.0, 3.0]));
    });

    test('subtracts mean and divides by scale', () {
      final params = ScalerParams(
        mean: Float64List.fromList([2.0, 4.0]),
        scale: Float64List.fromList([2.0, 2.0]),
      );
      // [6, 8] → [(6-2)/2, (8-4)/2] = [2, 2]
      final X = Float64List.fromList([6.0, 8.0]);
      final result = scaler.transform(X, 1, 2, params);
      expect(result[0], closeTo(2.0, 1e-10));
      expect(result[1], closeTo(2.0, 1e-10));
    });

    test('zero scale feature becomes 0', () {
      final params = ScalerParams(
        mean: Float64List.fromList([5.0]),
        scale: Float64List.fromList([0.0]),
      );
      final X = Float64List.fromList([100.0]);
      final result = scaler.transform(X, 1, 1, params);
      expect(result[0], equals(0.0));
    });

    test('multiple samples are scaled independently per-feature', () {
      final params = ScalerParams(
        mean: Float64List.fromList([0.0, 10.0]),
        scale: Float64List.fromList([1.0, 5.0]),
      );
      // sample0: [3, 20] → [3, 2]
      // sample1: [7, 0]  → [7, -2]
      final X = Float64List.fromList([3.0, 20.0, 7.0, 0.0]);
      final result = scaler.transform(X, 2, 2, params);
      expect(result[0], closeTo(3.0, 1e-10));
      expect(result[1], closeTo(2.0, 1e-10));
      expect(result[2], closeTo(7.0, 1e-10));
      expect(result[3], closeTo(-2.0, 1e-10));
    });

    test('does not modify original array', () {
      final params = ScalerParams(
        mean: Float64List.fromList([1.0]),
        scale: Float64List.fromList([2.0]),
      );
      final X = Float64List.fromList([5.0]);
      scaler.transform(X, 1, 1, params);
      expect(X[0], equals(5.0)); // unchanged
    });
  });
}
