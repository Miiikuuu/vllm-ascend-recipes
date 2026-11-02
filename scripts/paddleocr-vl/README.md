# PaddleOCR-VL extended functional checks

These optional helpers were extracted unchanged from the recipe's Verification
section. The page now contains only the tutorial-style OCR API smoke check.
These checks are functional regression tools, not OmniDocBench accuracy or the
ModelZoo document-pipeline performance benchmark.

From the repository root, with a running original PaddleOCR-VL server and Pillow
installed in the client environment:

```bash
python scripts/paddleocr-vl/prepare_samples.py --output samples
python scripts/paddleocr-vl/verify.py --base-url http://127.0.0.1:8000 --model PaddlePaddle/PaddleOCR-VL --samples samples --output request-results-canonical --warmup 0 --repeats 1
python scripts/paddleocr-vl/verify.py --base-url http://127.0.0.1:8000 --model PaddleOCR-VL-0.9B --samples samples --output request-results-alias --warmup 0 --repeats 1
```

Replace the server URL with the actual host and port. Preparation tries the
official ModelScope dataset and then the pinned Hugging Face dataset, validates
the original SHA256 values, and crops only input regions from demonstration
collages. For offline preparation, prefetch the four originals named in
`prepare_samples.py` into `samples/`, or use `--source-dir /path/to/originals`.
The same hashes remain required; no download or inference failure is ignored.

The verifier checks OCR, table, formula and chart content and `finish_reason`,
and saves raw API responses plus a summary. To make separate repeated latency
observations after the functional checks:

```bash
python scripts/paddleocr-vl/verify.py --base-url http://127.0.0.1:8000 --samples samples --output benchmark-results --warmup 1 --repeats 3
```

These latencies include API round trips at concurrency 1 and are not the
full-document benchmark. This extraction adds no new device-validation claim.
