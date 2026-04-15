set windows-shell := ["pwsh.exe", "/c"]

bench:
  dart run benchmark/rdst_benchmark.dart

train:
  Set-Location python; python3 create_integration_test_fixtures.py

test:
  dart test
