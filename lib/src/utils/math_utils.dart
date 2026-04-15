import 'dart:math' as math;
import 'dart:typed_data';

/// Small math utilities used throughout the RDST inference pipeline.
///
/// All functions are pure and operate on [Float64List] or plain [List<double>]
/// to avoid allocation where possible.

// ---------------------------------------------------------------------------
// Softmax / sigmoid
// ---------------------------------------------------------------------------

/// Computes the softmax of [scores], returning a new [Float64List].
///
/// Numerically stable: subtracts `max(scores)` before exponentiation.
Float64List softmax(Float64List scores) {
  double maxVal = double.negativeInfinity;
  for (final s in scores) {
    if (s > maxVal) maxVal = s;
  }
  final result = Float64List(scores.length);
  double sum = 0.0;
  for (var i = 0; i < scores.length; i++) {
    result[i] = math.exp(scores[i] - maxVal);
    sum += result[i];
  }
  for (var i = 0; i < scores.length; i++) {
    result[i] /= sum;
  }
  return result;
}

/// Computes the sigmoid of a scalar value.
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

/// Returns the [p]-th percentile (0–100) of [values] using linear
/// interpolation (matches NumPy's default).
///
/// [values] is not modified; a sorted copy is used internally.
double percentile(List<double> values, double p) {
  if (values.isEmpty) throw ArgumentError('values must not be empty');
  final sorted = List<double>.from(values)..sort();
  final n = sorted.length;
  final idx = (p / 100.0) * (n - 1);
  final lower = idx.floor();
  final upper = idx.ceil();
  if (lower == upper) return sorted[lower];
  final frac = idx - lower;
  return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

// ---------------------------------------------------------------------------
// Prime utilities
// ---------------------------------------------------------------------------

/// Returns `true` if [n] is prime.
bool isPrime(int n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if (n.isEven) return false;
  for (var i = 3; i * i <= n; i += 2) {
    if (n % i == 0) return false;
  }
  return true;
}

/// Returns all primes ≤ [limit] in ascending order.
List<int> primesUpTo(int limit) {
  final result = <int>[];
  for (var i = 2; i <= limit; i++) {
    if (isPrime(i)) result.add(i);
  }
  return result;
}
