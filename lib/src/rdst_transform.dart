import 'dart:typed_data';

import 'model/rdst_model.dart';
import 'utils/subsequence.dart';

/// Pure-Dart port of `RandomDilatedShapeletTransform.transform()` from aeon.
///
/// Given a set of time series `X` with shape `(nSamples, nChannels, nTimepoints)`,
/// produces a feature matrix of shape `(nSamples, 3 * nShapelets)`.
///
/// For each shapelet, three features are extracted per sample:
///   - `minDist`   : minimum L1 distance to any subsequence
///   - `argMin`    : index of the closest subsequence
///   - `occurrence`: count of subsequences closer than [ShapeletParams.threshold]
///
/// Shapelets are grouped by `(length, dilation)` so subsequences are computed
/// only once per unique pair, matching the optimisation in the Python code.
class RdstTransform {
  const RdstTransform();

  /// Transforms [X] using the fitted shapelets in [model].
  ///
  /// [X] is a flat [Float64List] of shape `(nSamples, nChannels, nTimepoints)`
  /// stored in sample-major, channel-major order:
  ///   `X[s, c, t] = X[s * nChannels * nTimepoints + c * nTimepoints + t]`
  ///
  /// Returns a [Float64List] of shape `(nSamples, 3 * nShapelets)` in row-major
  /// order.
  Float64List transform(
    Float64List X,
    int nSamples,
    int nChannels,
    int nTimepoints,
    RdstModel model,
  ) {
    final nShapelets = model.nShapelets;
    final nFeatures = 3 * nShapelets;
    final result = Float64List(nSamples * nFeatures);

    // Identify unique (length, dilation) pairs.
    final uniquePairs = <(int, int)>{};
    for (final s in model.shapelets) {
      uniquePairs.add((s.length, s.dilation));
    }

    for (var iSample = 0; iSample < nSamples; iSample++) {
      // Extract the single sample: shape (nChannels, nTimepoints)
      final sampleOffset = iSample * nChannels * nTimepoints;
      final sample = Float64List.sublistView(
          X, sampleOffset, sampleOffset + nChannels * nTimepoints);

      for (final pair in uniquePairs) {
        final (length, dilation) = pair;

        // All shapelets that share this (length, dilation).
        final groupShapelets = model.shapelets
            .asMap()
            .entries
            .where(
                (e) => e.value.length == length && e.value.dilation == dilation)
            .toList(growable: false);

        // Count non-normalised and normalised shapelets in this group.
        final nonNormEntries =
            groupShapelets.where((e) => !e.value.normalise).toList();
        final normEntries =
            groupShapelets.where((e) => e.value.normalise).toList();

        // Compute raw subsequences (needed for both norm and non-norm).
        final nSubs = nTimepoints - (length - 1) * dilation;
        final subs = getAllSubsequences(
            sample, nChannels, nTimepoints, length, dilation);

        // Non-normalised shapelets.
        for (final entry in nonNormEntries) {
          final iShp = entry.key;
          final shp = entry.value;
          final features = computeShapeletFeatures(
            subs,
            shp.values,
            shp.threshold,
            nSubs,
            nChannels,
            length,
          );
          final offset = iSample * nFeatures + 3 * iShp;
          result[offset] = features.minDist;
          result[offset + 1] = features.argMin;
          result[offset + 2] = features.occurrence;
        }

        // Normalised shapelets — compute sliding mean/std once for this pair.
        if (normEntries.isNotEmpty) {
          final (:means, :stds) =
              slidingMeanStd(sample, nChannels, nTimepoints, length, dilation);
          final normSubs = normaliseSubsequences(
              subs, means, stds, nSubs, nChannels, length);

          for (final entry in normEntries) {
            final iShp = entry.key;
            final shp = entry.value;
            final features = computeShapeletFeatures(
              normSubs,
              shp.values,
              shp.threshold,
              nSubs,
              nChannels,
              length,
            );
            final offset = iSample * nFeatures + 3 * iShp;
            result[offset] = features.minDist;
            result[offset + 1] = features.argMin;
            result[offset + 2] = features.occurrence;
          }
        }
      }
    }

    return result;
  }
}
