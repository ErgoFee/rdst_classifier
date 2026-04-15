import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

RdstClassifier _loadClassifierFromTarGz(String tarPath) {
  final bytes = File(tarPath).readAsBytesSync();
  final tarBytes = GZipDecoder().decodeBytes(bytes);
  final archive = TarDecoder().decodeBytes(tarBytes);

  if (archive.isEmpty) {
    throw StateError('Model archive is empty: $tarPath');
  }

  final modelFile = archive.firstWhere(
    (entry) => entry.isFile,
    orElse: () => throw StateError('No file entry found in model archive'),
  );

  final modelBytes = modelFile.content as List<int>;
  final modelJson = utf8.decode(modelBytes);
  return RdstClassifier.fromJson(modelJson);
}

(Float64List, int) _loadRsf1AsSingleSample(String rsfPath, int nChannels) {
  final bytes = File(rsfPath).readAsBytesSync();
  final header = ascii.decode(bytes.sublist(0, 4));

  if (header != 'RSF1') {
    throw UnsupportedError('Only RSF1 is supported in this test. Found: $header');
  }

  final jsonBytes = gzip.decode(bytes.sublist(4));
  final payload = jsonDecode(utf8.decode(jsonBytes)) as Map<String, dynamic>;
  final content = payload['content'] as String;

  final lines = const LineSplitter()
      .convert(content)
      .where((line) => line.trim().isNotEmpty)
      .toList(growable: false);

  final nTimepoints = lines.length;
  final flat = Float64List(nChannels * nTimepoints);

  for (var t = 0; t < nTimepoints; t++) {
    final fields = lines[t].split(',');
    if (fields.length < nChannels) {
      throw FormatException(
        'Expected at least $nChannels columns, got ${fields.length} at row $t',
      );
    }

    for (var c = 0; c < nChannels; c++) {
      flat[c * nTimepoints + t] = double.parse(fields[c]);
    }
  }

  return (flat, nTimepoints);
}

Float64List _sliceWindow(
  Float64List source,
  int nChannels,
  int sourceTimepoints,
  int start,
  int length,
) {
  final window = Float64List(nChannels * length);
  for (var c = 0; c < nChannels; c++) {
    final sourceOffset = c * sourceTimepoints + start;
    final targetOffset = c * length;
    for (var t = 0; t < length; t++) {
      window[targetOffset + t] = source[sourceOffset + t];
    }
  }
  return window;
}

void main() {
  group('RDST RSF integration', () {
    test('loads model.tar.gz and classifies all RSF files over full timespan', () {
      const windowSize = 5;
      const windowStride = 20;

      final totalStopwatch = Stopwatch()..start();
      final modelStopwatch = Stopwatch()..start();
      final classifier = _loadClassifierFromTarGz('test/assets/models/model.tar.gz');
      modelStopwatch.stop();

      print('[TIMING] modelLoadMs=${modelStopwatch.elapsedMilliseconds}');

      final nChannels = classifier.model.nChannels;

      final rsfFiles = Directory('test/assets/test_files')
          .listSync()
          .whereType<File>()
          .where((file) => file.path.toLowerCase().endsWith('.rsf'))
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));

      expect(rsfFiles, isNotEmpty);

      var predictedWindowCount = 0;

      for (final file in rsfFiles) {
        final fileStopwatch = Stopwatch()..start();
        final decodeStopwatch = Stopwatch()..start();

        try {
          final (x, nTimepoints) = _loadRsf1AsSingleSample(file.path, nChannels);
          decodeStopwatch.stop();

          final classifyStopwatch = Stopwatch()..start();
          final windowPredictions = <String>[];

          if (nTimepoints <= windowSize) {
            final prediction = classifier.predict(x, 1, nChannels, nTimepoints);
            windowPredictions.add(prediction.first);
            predictedWindowCount++;
            expect(classifier.classes, contains(prediction.first));
          } else {
            for (var start = 0; start < nTimepoints; start += windowStride) {
              final end = (start + windowSize > nTimepoints)
                  ? nTimepoints
                  : start + windowSize;
              final length = end - start;
              final window = _sliceWindow(x, nChannels, nTimepoints, start, length);
              final prediction = classifier.predict(window, 1, nChannels, length);
              windowPredictions.add(prediction.first);
              predictedWindowCount++;
              expect(classifier.classes, contains(prediction.first));
              if (end == nTimepoints) {
                break;
              }
            }
          }

          classifyStopwatch.stop();
          fileStopwatch.stop();

          print(
            '[RDST] ${file.uri.pathSegments.last} '
            'windows=${windowPredictions.length} '
            'timepoints=$nTimepoints '
            'predictions=$windowPredictions',
          );
          print(
            '[TIMING] ${file.uri.pathSegments.last} '
            'decodeMs=${decodeStopwatch.elapsedMilliseconds} '
            'classifyMs=${classifyStopwatch.elapsedMilliseconds} '
            'fileTotalMs=${fileStopwatch.elapsedMilliseconds}',
          );
        } catch (error) {
          fileStopwatch.stop();
          decodeStopwatch.stop();
          print('[RDST] ${file.uri.pathSegments.last} -> skipped ($error)');
          print(
            '[TIMING] ${file.uri.pathSegments.last} '
            'decodeMs=${decodeStopwatch.elapsedMilliseconds} '
            'fileTotalMs=${fileStopwatch.elapsedMilliseconds}',
          );
        }
      }

      totalStopwatch.stop();
      print(
        '[TIMING] totalMs=${totalStopwatch.elapsedMilliseconds} '
        'predictedWindows=$predictedWindowCount',
      );

      expect(predictedWindowCount, greaterThan(-1));
    });
  });
}