import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

void main() {
  // ---------------------------------------------------------------------------
  // getAllSubsequences
  // ---------------------------------------------------------------------------
  group('getAllSubsequences', () {
    test('single channel, length=3, dilation=1 — basic extraction', () {
      // X: 1 channel, 5 time points: [0,1,2,3,4]
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1);
      // nSubs = 5 - (3-1)*1 = 3
      expect(subs.length, equals(3 * 1 * 3));
      // sub 0: [0,1,2]
      expect(subs[0], equals(0.0));
      expect(subs[1], equals(1.0));
      expect(subs[2], equals(2.0));
      // sub 1: [1,2,3]
      expect(subs[3], equals(1.0));
      expect(subs[4], equals(2.0));
      expect(subs[5], equals(3.0));
      // sub 2: [2,3,4]
      expect(subs[6], equals(2.0));
      expect(subs[7], equals(3.0));
      expect(subs[8], equals(4.0));
    });

    test('dilation=2 skips every other element', () {
      // X: 1 channel, 5 time points: [0,1,2,3,4]
      // length=2, dilation=2 → nSubs = 5 - (2-1)*2 = 3
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 2, 2);
      expect(subs.length, equals(3 * 1 * 2));
      // sub 0: [x[0], x[2]] = [0, 2]
      expect(subs[0], equals(0.0));
      expect(subs[1], equals(2.0));
      // sub 1: [x[1], x[3]] = [1, 3]
      expect(subs[2], equals(1.0));
      expect(subs[3], equals(3.0));
      // sub 2: [x[2], x[4]] = [2, 4]
      expect(subs[4], equals(2.0));
      expect(subs[5], equals(4.0));
    });

    test('multi-channel subsequence layout', () {
      // X: 2 channels, 4 time points
      // ch0: [10,20,30,40], ch1: [1,2,3,4]
      final X =
          Float64List.fromList([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]);
      // length=2, dilation=1 → nSubs = 3
      final subs = getAllSubsequences(X, 2, 4, 2, 1);
      expect(subs.length, equals(3 * 2 * 2));

      // sub0: ch0=[10,20], ch1=[1,2]
      expect(subs[0], equals(10.0)); // sub0, ch0, j=0
      expect(subs[1], equals(20.0)); // sub0, ch0, j=1
      expect(subs[2], equals(1.0)); //  sub0, ch1, j=0
      expect(subs[3], equals(2.0)); //  sub0, ch1, j=1

      // sub1: ch0=[20,30], ch1=[2,3]
      expect(subs[4], equals(20.0));
      expect(subs[5], equals(30.0));
      expect(subs[6], equals(2.0));
      expect(subs[7], equals(3.0));
    });
  });

  // ---------------------------------------------------------------------------
  // slidingMeanStd
  // ---------------------------------------------------------------------------
  group('slidingMeanStd', () {
    test('constant series has std=0', () {
      final X = Float64List.fromList([3.0, 3.0, 3.0, 3.0, 3.0]);
      final (:means, :stds) = slidingMeanStd(X, 1, 5, 3, 1);
      // nSubs = 3
      for (var i = 0; i < 3; i++) {
        expect(means[i], closeTo(3.0, 1e-10));
        expect(stds[i], closeTo(0.0, 1e-10));
      }
    });

    test('mean is correct for simple sequence', () {
      // ch0: [0,1,2,3,4], length=3, dilation=1
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final (:means, :stds) = slidingMeanStd(X, 1, 5, 3, 1);
      // nSubs=3: means=[1,2,3]
      expect(means[0], closeTo(1.0, 1e-10));
      expect(means[1], closeTo(2.0, 1e-10));
      expect(means[2], closeTo(3.0, 1e-10));
    });

    test('std is correct for known sequence', () {
      // [0,1,2,3,4], length=3, window [0,1,2]: mean=1, var=2/3, std=sqrt(2/3)
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final (:means, :stds) = slidingMeanStd(X, 1, 5, 3, 1);
      final expectedStd0 = 0.816496580927726; // sqrt(2/3)
      expect(stds[0], closeTo(expectedStd0, 1e-6));
    });

    test('multi-channel', () {
      // ch0: [0,2,4,6], ch1: [1,1,1,1], length=2, dilation=1 → nSubs=3
      final X = Float64List.fromList([0.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0, 1.0]);
      final (:means, :stds) = slidingMeanStd(X, 2, 4, 2, 1);
      // ch0 means: [1, 3, 5]
      expect(means[0 * 3 + 0], closeTo(1.0, 1e-10)); // ch0, sub0
      expect(means[0 * 3 + 1], closeTo(3.0, 1e-10)); // ch0, sub1
      expect(means[0 * 3 + 2], closeTo(5.0, 1e-10)); // ch0, sub2
      // ch1 means: [1,1,1]
      expect(means[1 * 3 + 0], closeTo(1.0, 1e-10));
      // ch1 std: 0 (constant)
      expect(stds[1 * 3 + 0], closeTo(0.0, 1e-10));
    });
  });

  // ---------------------------------------------------------------------------
  // normaliseSubsequences
  // ---------------------------------------------------------------------------
  group('normaliseSubsequences', () {
    test('normalised subs have zero mean and unit std', () {
      // Build a simple case
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1); // nSubs=3
      final (:means, :stds) = slidingMeanStd(X, 1, 5, 3, 1);
      final normed = normaliseSubsequences(subs, means, stds, 3, 1, 3);

      for (var i = 0; i < 3; i++) {
        if (stds[i] > 1e-8) {
          // compute mean and std of normed sub i
          double sumV = 0, sumV2 = 0;
          for (var j = 0; j < 3; j++) {
            sumV += normed[i * 3 + j];
            sumV2 += normed[i * 3 + j] * normed[i * 3 + j];
          }
          final m = sumV / 3;
          final s = (sumV2 / 3 - m * m);
          expect(m.abs(), lessThan(1e-10));
          expect(s, closeTo(1.0, 1e-6));
        }
      }
    });

    test('zero-std channels are zeroed out', () {
      // constant channel → std=0 → normalised sub should be 0
      final X = Float64List.fromList([5.0, 5.0, 5.0, 5.0, 5.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1);
      final (:means, :stds) = slidingMeanStd(X, 1, 5, 3, 1);
      final normed = normaliseSubsequences(subs, means, stds, 3, 1, 3);
      for (final v in normed) {
        expect(v, equals(0.0));
      }
    });
  });

  // ---------------------------------------------------------------------------
  // computeShapeletFeatures
  // ---------------------------------------------------------------------------
  group('computeShapeletFeatures', () {
    test('minDist is zero when shapelet equals a subsequence', () {
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1); // nSubs=3
      // Shapelet = [1,2,3] (second sub)
      final shpValues = Float64List.fromList([1.0, 2.0, 3.0]);
      final result = computeShapeletFeatures(subs, shpValues, 1.0, 3, 1, 3);
      expect(result.minDist, closeTo(0.0, 1e-10));
      expect(result.argMin, equals(1.0));
    });

    test('occurrence counts subs within threshold', () {
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1);
      // Shapelet = [1,2,3]; threshold=2 → subs within L1 dist 2:
      //   sub0=[0,1,2]: dist=|0-1|+|1-2|+|2-3|=3 → no
      //   sub1=[1,2,3]: dist=0 → yes
      //   sub2=[2,3,4]: dist=3 → no
      final shpValues = Float64List.fromList([1.0, 2.0, 3.0]);
      final result = computeShapeletFeatures(subs, shpValues, 2.0, 3, 1, 3);
      expect(result.occurrence, equals(1.0));
    });

    test('occurrence is zero when threshold very small', () {
      final X = Float64List.fromList([0.0, 1.0, 2.0, 3.0, 4.0]);
      final subs = getAllSubsequences(X, 1, 5, 3, 1);
      final shpValues = Float64List.fromList([1.0, 2.0, 3.0]);
      final result = computeShapeletFeatures(subs, shpValues, 0.0, 3, 1, 3);
      // Even the exact match has dist=0 which is NOT < 0
      expect(result.occurrence, equals(0.0));
    });
  });
}
