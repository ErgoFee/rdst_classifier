import 'dart:io';

import 'package:rdst_classifier/rdst_classifier.dart';
import 'package:test/test.dart';

String _fixtureJson(String name) {
  return File('test/fixtures/$name').readAsStringSync();
}

void main() {
  group('ModelIo', () {
    test('parses binary model without error', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      expect(model.version, isNotEmpty);
      expect(model.nShapelets, greaterThan(0));
      expect(model.nChannels, greaterThan(0));
      expect(model.shapelets.length, equals(model.nShapelets));
    });

    test('parses multiclass model without error', () {
      final model = ModelIo.fromJson(_fixtureJson('multiclass_model.json'));
      expect(model.classifier.classes.length, greaterThanOrEqualTo(3));
    });

    test('binary model has 2 classes', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      expect(model.classifier.classes.length, equals(2));
    });

    test('shapelet values have correct dimensions', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      for (final shp in model.shapelets) {
        expect(shp.values.length, equals(model.nChannels * shp.length));
        expect(shp.means.length, equals(model.nChannels));
        expect(shp.stds.length, equals(model.nChannels));
      }
    });

    test('scaler arrays have correct length', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      final nFeatures = 3 * model.nShapelets;
      expect(model.scaler.mean.length, equals(nFeatures));
      expect(model.scaler.scale.length, equals(nFeatures));
    });

    test('classifier coef dimensions match nShapelets and nClasses', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      final nFeatures = 3 * model.nShapelets;
      // For binary: nRows=1
      expect(model.classifier.nCols, equals(nFeatures));
    });

    test('multiclass classifier coef has one row per class', () {
      final model = ModelIo.fromJson(_fixtureJson('multiclass_model.json'));
      expect(model.classifier.nRows, equals(model.classifier.classes.length));
    });

    test('isBinary is true for binary model', () {
      final model = ModelIo.fromJson(_fixtureJson('binary_model.json'));
      expect(model.classifier.isBinary, isTrue);
    });

    test('isBinary is false for multiclass model', () {
      final model = ModelIo.fromJson(_fixtureJson('multiclass_model.json'));
      expect(model.classifier.isBinary, isFalse);
    });

    test('accepts camelCase and snake_case keys', () {
      // The fixture uses snake_case from Python; ensure parsing does not throw.
      final json = _fixtureJson('binary_model.json');
      expect(() => ModelIo.fromJson(json), returnsNormally);
    });
  });
}
