# Changelog

## [1.0.2](https://github.com/ffreis-org/ffreis-rust-onnx-model-serving/compare/v1.0.1...v1.0.2) (2026-07-16)


### Bug Fixes

* **ci:** pin ffreis-workflows-general to v1.7.0 ([#89](https://github.com/ffreis-org/ffreis-rust-onnx-model-serving/issues/89)) ([6c1e578](https://github.com/ffreis-org/ffreis-rust-onnx-model-serving/commit/6c1e5789458056207cb4535ba46ef68fcbd3d165))
* **security:** apt-get upgrade + non-root USER in Dockerfile ([#88](https://github.com/ffreis-org/ffreis-rust-onnx-model-serving/issues/88)) ([0b39280](https://github.com/ffreis-org/ffreis-rust-onnx-model-serving/commit/0b39280954f41c4affc7ba44c146d810da2ca664))

## [1.0.1](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/compare/v1.0.0...v1.0.1) (2026-06-13)


### Bug Fixes

* correct sonar projectKey to ffreis-rust-onnx-model-serving ([#72](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/72)) ([f5a73c4](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/f5a73c4e3917d4b12338c008cf82074ba6807f96))
* **grype:** bump workflows-general SHA to prevent self-scan CVEs ([#71](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/71)) ([ceebc8b](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/ceebc8b36939b69272f1fcfad15a021ebf17c0ac))

## 1.0.0 (2026-05-26)


### Features

* cargo updates ([#29](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/29)) ([d6faf2a](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/d6faf2ab9220f542430319558cbc10ff172b6343))
* **deps:** migrate to ffreis-platform-standards ([#40](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/40)) ([4a08776](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/4a087769e45d794447a6ec5d8f27fc9003a20c74))
* first commit, contains docker build, makefile, dockerfile and whatnot for a simple rust builder and runner ([05e1b87](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/05e1b87986f12e35ca9cf5b6761f86b27e2da7e6))
* improve sonar, trivy and ci in general ([#13](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/13)) ([97c5c7f](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/97c5c7fefe87164ef49ab497c49b35757e4ec257))
* onnx api and grpc ([#20](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/20)) ([086ee1b](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/086ee1bbdd5661b0599ee6d89ce0518c087e5607))
* push work so far to main  ([#1](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/1)) ([3b9b5d5](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/3b9b5d582d1567fd322a5fd07d24120880ae4a69))


### Bug Fixes

* correct readme ([8e369af](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/8e369af966659dc4c92f2eb3a2cea33d55ec2617))
* improve sonar and change image names ([#11](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/11)) ([20f69c4](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/20f69c461d62dcc7f53335ea1b8fd43e95d6fae8))
* improve trivy ([#15](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/15)) ([bef61f8](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/bef61f85afab62b10b8f4dfe2ca7ac80b48b8228))
* parallel trivy ([#18](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/issues/18)) ([b33a857](https://github.com/FelipeFuhr/ffreis-rust-onnx-model-serving/commit/b33a857b5aacbb109bf70f27d90efeadff08caae))
