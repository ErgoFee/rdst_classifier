// ignore_for_file: avoid_print
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:rdst_classifier/rdst_classifier.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
String _fixtureJson(String name) =>
    File('test/fixtures/$name').readAsStringSync();

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

/// Tile [src] (length = srcSamples * stride) to reach exactly [targetSamples].
Float64List _tile(
    Float64List src, int srcSamples, int targetSamples, int stride) {
  final result = Float64List(targetSamples * stride);
  for (var i = 0; i < targetSamples; i++) {
    final srcIdx = (i % srcSamples) * stride;
    result.setRange(i * stride, i * stride + stride, src, srcIdx);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Statistics helper
// ---------------------------------------------------------------------------
({double mean, double min, double max, double p50, double p95, double p99})
    _stats(List<double> values) {
  final sorted = [...values]..sort();
  final n = sorted.length;
  double sum = 0;
  for (final v in sorted) sum += v;
  return (
    mean: sum / n,
    min: sorted.first,
    max: sorted.last,
    p50: sorted[(n * 0.50).floor()],
    p95: sorted[(n * 0.95).floor()],
    p99: sorted[(n * 0.99).floor()],
  );
}

void _printStats(String label, List<double> usPerSample) {
  final s = _stats(usPerSample);
  print(
    '$label: '
    'mean=${s.mean.toStringAsFixed(1)} µs  '
    'min=${s.min.toStringAsFixed(1)}  '
    'p50=${s.p50.toStringAsFixed(1)}  '
    'p95=${s.p95.toStringAsFixed(1)}  '
    'p99=${s.p99.toStringAsFixed(1)}  '
    'max=${s.max.toStringAsFixed(1)}',
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
void main() {
  const targetBatch = 1000; // samples per batch run
  const batchRuns = 10; // how many times to run the 1000-sample batch
  const singleRuns = 1000; // individual single-sample timings

  // ── Load model ────────────────────────────────────────────────────────────
  print('Loading model…');
  final clf = RdstClassifier.fromJson(_fixtureJson('integration_model.json'));
  print('  classes : ${clf.classes}');

  // ── Load + tile fixture data to 1000 samples ──────────────────────────────
  final preds = jsonDecode(_fixtureJson('integration_predictions.json'))
      as Map<String, dynamic>;
  final rawX = preds['test_X'] as List<dynamic>;
  final srcSamples = rawX.length; // 574
  final nChannels = (rawX[0] as List<dynamic>).length;
  final nTimepoints = ((rawX[0] as List<dynamic>)[0] as List<dynamic>).length;
  final stride = nChannels * nTimepoints;

  final srcFlat = _toFlat(rawX);
  final batchX = _tile(srcFlat, srcSamples, targetBatch, stride);

  print('  input   : $srcSamples fixture samples tiled to $targetBatch');
  print('  shape   : nChannels=$nChannels  nTimepoints=$nTimepoints');
  print('');

  // ── Warm-up (JIT) ─────────────────────────────────────────────────────────
  print('Warming up (5 batch runs)…');
  for (var i = 0; i < 5; i++) {
    clf.predict(batchX, targetBatch, nChannels, nTimepoints);
  }
  print('');

  // ══════════════════════════════════════════════════════════════════════════
  // BENCHMARK 1: batch of 1000 samples — predict()
  // ══════════════════════════════════════════════════════════════════════════
  print('── Batch predict() · $batchRuns runs of $targetBatch samples ──');
  final batchPredictUs = <double>[];
  for (var r = 0; r < batchRuns; r++) {
    final sw = Stopwatch()..start();
    clf.predict(batchX, targetBatch, nChannels, nTimepoints);
    sw.stop();
    final totalUs = sw.elapsedMicroseconds.toDouble();
    batchPredictUs.add(totalUs / targetBatch);
    final runLabel = '${r + 1}'.padLeft(2);
    print(
      '  run $runLabel: total=${(totalUs / 1000).toStringAsFixed(1)} ms  '
      'per-sample=${(totalUs / targetBatch).toStringAsFixed(1)} µs',
    );
  }
  print('');
  _printStats('predict()  per-sample', batchPredictUs);
  print('');

  // ══════════════════════════════════════════════════════════════════════════
  // BENCHMARK 2: batch of 1000 samples — predictProba()
  // ══════════════════════════════════════════════════════════════════════════
  print('── Batch predictProba() · $batchRuns runs of $targetBatch samples ──');
  final batchProbaUs = <double>[];
  for (var r = 0; r < batchRuns; r++) {
    final sw = Stopwatch()..start();
    clf.predictProba(batchX, targetBatch, nChannels, nTimepoints);
    sw.stop();
    final totalUs = sw.elapsedMicroseconds.toDouble();
    batchProbaUs.add(totalUs / targetBatch);
    final runLabel2 = '${r + 1}'.padLeft(2);
    print(
      '  run $runLabel2: total=${(totalUs / 1000).toStringAsFixed(1)} ms  '
      'per-sample=${(totalUs / targetBatch).toStringAsFixed(1)} µs',
    );
  }
  print('');
  _printStats('predictProba() per-sample', batchProbaUs);
  print('');

  // ══════════════════════════════════════════════════════════════════════════
  // BENCHMARK 3: individual single-sample calls — predict()
  // ══════════════════════════════════════════════════════════════════════════
  print('── Single-sample predict() · $singleRuns individual calls ──');
  final singlePredictUs = <double>[];
  for (var i = 0; i < singleRuns; i++) {
    final offset = (i % srcSamples) * stride;
    final singleX = Float64List.sublistView(srcFlat, offset, offset + stride);
    final sw = Stopwatch()..start();
    clf.predict(singleX, 1, nChannels, nTimepoints);
    sw.stop();
    singlePredictUs.add(sw.elapsedMicroseconds.toDouble());
  }
  _printStats('single predict()       ', singlePredictUs);
  print('');

  // ══════════════════════════════════════════════════════════════════════════
  // BENCHMARK 4: individual single-sample calls — predictProba()
  // ══════════════════════════════════════════════════════════════════════════
  print('── Single-sample predictProba() · $singleRuns individual calls ──');
  final singleProbaUs = <double>[];
  for (var i = 0; i < singleRuns; i++) {
    final offset = (i % srcSamples) * stride;
    final singleX = Float64List.sublistView(srcFlat, offset, offset + stride);
    final sw = Stopwatch()..start();
    clf.predictProba(singleX, 1, nChannels, nTimepoints);
    sw.stop();
    singleProbaUs.add(sw.elapsedMicroseconds.toDouble());
  }
  _printStats('single predictProba()  ', singleProbaUs);
  print('');

  // ── Summary ───────────────────────────────────────────────────────────────
  final bpS = _stats(batchPredictUs);
  final bpaS = _stats(batchProbaUs);
  final spS = _stats(singlePredictUs);
  final spaS = _stats(singleProbaUs);

  print('══════════════ SUMMARY ══════════════');
  print(
      'Model: ${clf.classes.length} classes, ${nChannels} channels, ${nTimepoints} timepoints');
  print('');
  print('                        mean      p50      p95');
  print(
    'batch  predict()    ${bpS.mean.toStringAsFixed(1).padLeft(7)} µs  '
    '${bpS.p50.toStringAsFixed(1).padLeft(7)} µs  '
    '${bpS.p95.toStringAsFixed(1).padLeft(7)} µs',
  );
  print(
    'batch  predictProba ${bpaS.mean.toStringAsFixed(1).padLeft(7)} µs  '
    '${bpaS.p50.toStringAsFixed(1).padLeft(7)} µs  '
    '${bpaS.p95.toStringAsFixed(1).padLeft(7)} µs',
  );
  print(
    'single predict()    ${spS.mean.toStringAsFixed(1).padLeft(7)} µs  '
    '${spS.p50.toStringAsFixed(1).padLeft(7)} µs  '
    '${spS.p95.toStringAsFixed(1).padLeft(7)} µs',
  );
  print(
    'single predictProba ${spaS.mean.toStringAsFixed(1).padLeft(7)} µs  '
    '${spaS.p50.toStringAsFixed(1).padLeft(7)} µs  '
    '${spaS.p95.toStringAsFixed(1).padLeft(7)} µs',
  );
  print('═════════════════════════════════════');
}
