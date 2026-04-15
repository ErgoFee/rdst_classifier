import 'dart:typed_data';

/// Parameters for a single dilated shapelet.
///
/// [values] is a 2D array of shape `(nChannels, length)` stored as a
/// flat [Float64List] in row-major order (channel-major).
///
/// The helper [valueAt] provides indexed access:
/// `valueAt(channel, i) == values[channel * length + i]`
class ShapeletParams {
  /// Flat (channel-major) shapelet values: length = nChannels * length.
  final Float64List values;

  /// Number of channels this shapelet was extracted from.
  final int nChannels;

  /// Number of time points in the shapelet (the "window" length).
  final int length;

  /// Dilation factor used when extracting subsequences.
  final int dilation;

  /// Distance threshold for computing shapelet occurrence count.
  final double threshold;

  /// Whether this shapelet requires z-normalised subsequences.
  final bool normalise;

  /// Per-channel mean of the shapelet at training time.
  /// Shape: `(nChannels,)`.  Not used during inference.
  final Float64List means;

  /// Per-channel standard deviation of the shapelet at training time.
  /// Shape: `(nChannels,)`.  Not used during inference.
  final Float64List stds;

  const ShapeletParams({
    required this.values,
    required this.nChannels,
    required this.length,
    required this.dilation,
    required this.threshold,
    required this.normalise,
    required this.means,
    required this.stds,
  });

  /// Returns the value at [channel] and position [i] within the shapelet.
  double valueAt(int channel, int i) => values[channel * length + i];
}
