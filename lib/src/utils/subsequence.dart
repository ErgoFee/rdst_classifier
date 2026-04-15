import 'dart:typed_data';

/// Threshold below which a standard deviation is treated as zero during
/// normalisation, matching `AEON_NUMBA_STD_THRESHOLD = 1e-8`.
const double _kStdThreshold = 1e-8;

// ---------------------------------------------------------------------------
// Subsequence extraction
// ---------------------------------------------------------------------------

/// Returns all dilated subsequences of a single time series [X].
///
/// [X] has shape `(nChannels, nTimepoints)` stored as a flat [Float64List]
/// in **channel-major** order: `X[channel, t] = X[channel * nTimepoints + t]`.
///
/// Returns a 3-D Float64List of shape `(nSubs, nChannels, length)` in
/// **sub-major** order:
///   `result[sub, channel, i] = result[sub * nChannels * length + channel * length + i]`
///
/// Number of subsequences: `nSubs = nTimepoints - (length - 1) * dilation`.
Float64List getAllSubsequences(
  Float64List X,
  int nChannels,
  int nTimepoints,
  int length,
  int dilation,
) {
  final nSubs = nTimepoints - (length - 1) * dilation;
  assert(nSubs > 0,
      'No valid subsequences for length=$length dilation=$dilation nTimepoints=$nTimepoints');
  final result = Float64List(nSubs * nChannels * length);
  for (var iSub = 0; iSub < nSubs; iSub++) {
    for (var c = 0; c < nChannels; c++) {
      for (var j = 0; j < length; j++) {
        final tIdx = iSub + j * dilation;
        result[iSub * nChannels * length + c * length + j] =
            X[c * nTimepoints + tIdx];
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Sliding mean and standard deviation
// ---------------------------------------------------------------------------

/// Computes the per-channel sliding mean and standard deviation for all dilated
/// subsequences of [X].
///
/// [X] has shape `(nChannels, nTimepoints)` stored channel-major in [Float64List].
///
/// Returns a record `(means, stds)` each of shape `(nChannels, nSubs)` stored
/// channel-major:
///   `means[channel, iSub] = means[channel * nSubs + iSub]`
({Float64List means, Float64List stds}) slidingMeanStd(
  Float64List X,
  int nChannels,
  int nTimepoints,
  int length,
  int dilation,
) {
  final nSubs = nTimepoints - (length - 1) * dilation;
  final means = Float64List(nChannels * nSubs);
  final stds = Float64List(nChannels * nSubs);

  for (var iModDil = 0; iModDil < dilation; iModDil++) {
    // Build the index array for the first subsequence starting at iModDil.
    final idxSub = List<int>.generate(length, (i) => i * dilation + iModDil);

    // Check first subsequence is valid.
    if (idxSub.last >= nTimepoints) continue;

    final sum = Float64List(nChannels);
    final sum2 = Float64List(nChannels);

    // Initialise sums for first subsequence.
    for (var j = 0; j < length; j++) {
      for (var c = 0; c < nChannels; c++) {
        final v = X[c * nTimepoints + idxSub[j]];
        sum[c] += v;
        sum2[c] += v * v;
      }
    }

    // Write first subsequence statistics.
    _writeMeanStd(means, stds, sum, sum2, nChannels, nSubs, iModDil, length);

    // Slide forward.
    for (var iSubStart = iModDil + dilation;
        iSubStart < nSubs;
        iSubStart += dilation) {
      final newIdx = idxSub.last + dilation;
      final oldIdx = idxSub.first;

      if (newIdx >= nTimepoints) break;

      for (var c = 0; c < nChannels; c++) {
        final vNew = X[c * nTimepoints + newIdx];
        final vOld = X[c * nTimepoints + oldIdx];
        sum[c] += vNew - vOld;
        sum2[c] += vNew * vNew - vOld * vOld;
      }
      _writeMeanStd(
          means, stds, sum, sum2, nChannels, nSubs, iSubStart, length);

      // Advance index array.
      for (var j = 0; j < length; j++) {
        idxSub[j] += dilation;
      }
    }
  }

  return (means: means, stds: stds);
}

void _writeMeanStd(
  Float64List means,
  Float64List stds,
  Float64List sum,
  Float64List sum2,
  int nChannels,
  int nSubs,
  int iSub,
  int length,
) {
  for (var c = 0; c < nChannels; c++) {
    final m = sum[c] / length;
    means[c * nSubs + iSub] = m;
    final variance = sum2[c] / length - m * m;
    if (variance > _kStdThreshold) {
      stds[c * nSubs + iSub] = variance >= 0 ? variance.sqrt() : 0.0;
    }
    // else std stays 0.0 (already initialised)
  }
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

/// Z-normalises the subsequences in [subs] in-place using [means] and [stds].
///
/// [subs] has shape `(nSubs, nChannels, length)` stored sub-major.
/// [means] and [stds] each have shape `(nChannels, nSubs)` stored channel-major.
///
/// Returns a new normalised array (does not modify [subs]).
Float64List normaliseSubsequences(
  Float64List subs,
  Float64List means,
  Float64List stds,
  int nSubs,
  int nChannels,
  int length,
) {
  final result = Float64List(subs.length);
  for (var iSub = 0; iSub < nSubs; iSub++) {
    for (var c = 0; c < nChannels; c++) {
      final std = stds[c * nSubs + iSub];
      if (std > _kStdThreshold) {
        final mean = means[c * nSubs + iSub];
        for (var j = 0; j < length; j++) {
          result[iSub * nChannels * length + c * length + j] =
              (subs[iSub * nChannels * length + c * length + j] - mean) / std;
        }
      }
      // else: result stays 0.0 which is the correct normalisation when std=0
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Shapelet features
// ---------------------------------------------------------------------------

/// Computes the three shapelet features `(minDist, argMin, occurrence)` for a
/// single shapelet applied to all subsequences.
///
/// [subs] has shape `(nSubs, nChannels, length)` stored sub-major.
/// [shpValues] has shape `(nChannels, length)` stored channel-major in a
///   flat [Float64List] (e.g. [ShapeletParams.values]).
/// [threshold] is the shapelet distance threshold.
///
/// Returns `(minDist, argMin, occurrenceCount)` as doubles matching the Python
/// implementation.
({double minDist, double argMin, double occurrence}) computeShapeletFeatures(
  Float64List subs,
  Float64List shpValues,
  double threshold,
  int nSubs,
  int nChannels,
  int length,
) {
  var minDist = double.infinity;
  var argMin = double.infinity;
  var so = 0.0;

  for (var iSub = 0; iSub < nSubs; iSub++) {
    var dist = 0.0;
    for (var c = 0; c < nChannels; c++) {
      for (var j = 0; j < length; j++) {
        dist += (subs[iSub * nChannels * length + c * length + j] -
                shpValues[c * length + j])
            .abs();
      }
    }
    if (dist < minDist) {
      minDist = dist;
      argMin = iSub.toDouble();
    }
    if (dist < threshold) {
      so += 1.0;
    }
  }

  return (minDist: minDist, argMin: argMin, occurrence: so);
}

extension on double {
  double sqrt() {
    // dart:math sqrt
    if (this < 0) return 0.0;
    return _sqrt(this);
  }
}

double _sqrt(double x) {
  if (x == 0.0) return 0.0;
  // Newton–Raphson
  var r = x;
  var prev = 0.0;
  while (r != prev) {
    prev = r;
    r = (r + x / r) * 0.5;
  }
  return r;
}
