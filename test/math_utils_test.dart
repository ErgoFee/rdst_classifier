import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

void main() {
  group('softmax', () {
    test('sums to 1', () {
      final result = softmax(Float64List.fromList([1.0, 2.0, 3.0]));
      expect(result.fold(0.0, (a, b) => a + b), closeTo(1.0, 1e-10));
    });

    test('largest input gets largest probability', () {
      final result = softmax(Float64List.fromList([1.0, 5.0, 2.0]));
      expect(result[1], greaterThan(result[0]));
      expect(result[1], greaterThan(result[2]));
    });

    test('uniform inputs → uniform output', () {
      final result = softmax(Float64List.fromList([2.0, 2.0, 2.0]));
      for (final p in result) {
        expect(p, closeTo(1.0 / 3.0, 1e-10));
      }
    });

    test('numerically stable with large values', () {
      final result = softmax(Float64List.fromList([1000.0, 1001.0, 1002.0]));
      expect(result.fold(0.0, (a, b) => a + b), closeTo(1.0, 1e-10));
      expect(result[2], greaterThan(result[1]));
    });

    test('single element → 1.0', () {
      final result = softmax(Float64List.fromList([42.0]));
      expect(result[0], closeTo(1.0, 1e-10));
    });
  });

  group('sigmoid', () {
    test('sigmoid(0) == 0.5', () {
      expect(sigmoid(0.0), closeTo(0.5, 1e-10));
    });

    test('sigmoid(large positive) ≈ 1', () {
      expect(sigmoid(100.0), closeTo(1.0, 1e-10));
    });

    test('sigmoid(large negative) ≈ 0', () {
      expect(sigmoid(-100.0), closeTo(0.0, 1e-10));
    });

    test('sigmoid is monotonically increasing', () {
      for (var x = -5.0; x < 5.0; x += 0.5) {
        expect(sigmoid(x + 0.5), greaterThan(sigmoid(x)));
      }
    });

    test('sigmoid(-x) == 1 - sigmoid(x)', () {
      for (final x in [-3.0, -1.0, 0.5, 2.0]) {
        expect(sigmoid(-x), closeTo(1.0 - sigmoid(x), 1e-10));
      }
    });
  });

  group('percentile', () {
    test('0th percentile is min', () {
      expect(percentile([3.0, 1.0, 2.0], 0), equals(1.0));
    });

    test('100th percentile is max', () {
      expect(percentile([3.0, 1.0, 2.0], 100), equals(3.0));
    });

    test('50th percentile of [1,2,3] is 2', () {
      expect(percentile([1.0, 2.0, 3.0], 50), closeTo(2.0, 1e-10));
    });

    test('linear interpolation', () {
      // [0,1,2,3] — 25th percentile should be 0.75
      final result = percentile([0.0, 1.0, 2.0, 3.0], 25);
      expect(result, closeTo(0.75, 1e-10));
    });

    test('throws on empty list', () {
      expect(() => percentile([], 50), throwsArgumentError);
    });
  });

  group('isPrime', () {
    test('returns false for n < 2', () {
      expect(isPrime(-1), isFalse);
      expect(isPrime(0), isFalse);
      expect(isPrime(1), isFalse);
    });

    test('2 is prime', () => expect(isPrime(2), isTrue));
    test('3 is prime', () => expect(isPrime(3), isTrue));
    test('4 is not prime', () => expect(isPrime(4), isFalse));
    test('17 is prime', () => expect(isPrime(17), isTrue));
    test('100 is not prime', () => expect(isPrime(100), isFalse));
  });

  group('primesUpTo', () {
    test('primes up to 10', () {
      expect(primesUpTo(10), equals([2, 3, 5, 7]));
    });

    test('primes up to 1 is empty', () {
      expect(primesUpTo(1), isEmpty);
    });

    test('primes up to 2', () {
      expect(primesUpTo(2), equals([2]));
    });

    test('all returned values are prime', () {
      for (final p in primesUpTo(50)) {
        expect(isPrime(p), isTrue);
      }
    });
  });
}
