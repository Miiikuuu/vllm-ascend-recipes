import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

REVISION = '702b96b0a06740b0ed3d41ba382a5bf69986dcae'
SAMPLES = {
    'ocr': ('text_english_arabic.jpg', (45, 159, 607, 449), 'f670627f3afa2e7ea312c71ce5c3c7b77a42938f9a1e62e5ba2505e19f5dede7'),
    'table': ('table_01.jpg', (713, 81, 1302, 424), 'ca798601c74d70cbb34a3a0b91161d21b79513bb049791c96dfed14bb087ceaf'),
    'formula': ('formula_EN.jpg', (126, 1328, 594, 1570), 'b9a96b4be44896a45df21103ba1b06af18621381b563707a42fa6cdf21f837d1'),
    'chart': ('chart_01.jpg', (27, 81, 614, 400), 'd77acb00585d16f978123bc8288c41e7a589083df51e18076a20853892c0d4a9'),
}


def download_source(urls):
    failures = []
    for url in urls:
        for attempt in range(1, 4):
            print(f'Sample download {attempt}/3: GET {url}', flush=True)
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    return response.read()
            except OSError as error:
                retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == 3:
                    failures.append(f'GET {url}: {error}')
                    print(f'Sample source unavailable: {failures[-1]}', flush=True)
                    break
                delay = 2 ** attempt
                print(f'Sample download failed: GET {url}: {error}; retrying in {delay}s', flush=True)
                time.sleep(delay)
    raise OSError(
        f'Sample download failed from all official sources: {"; ".join(failures)}. '
        'Prefetch the same pinned original into samples/ on a network-enabled client; '
        'the SHA256 check remains required.'
    )


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('samples'))
parser.add_argument('--source-dir', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest = []
for task, (filename, box, expected_hash) in SAMPLES.items():
    source = (args.source_dir or args.output) / filename
    url = '/'.join(('https://huggingface.co', 'datasets', 'PaddlePaddle', 'PaddleOCR-VL_demo', 'resolve', REVISION, 'imgs', filename))
    if not source.exists():
        # Both official dataset copies must match the immutable source hash below.
        modelscope_url = '/'.join(('https://modelscope.cn', 'datasets', 'PaddlePaddle', 'PaddleOCR-VL_demo', 'resolve', 'master', 'imgs', filename))
        data = download_source((modelscope_url, url))
        assert hashlib.sha256(data).hexdigest() == expected_hash, f'Source mismatch: {filename}'
        source.write_bytes(data)
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash, f'Source mismatch: {filename}'
    # The official demo collages contain input and expected output panels.
    # Crop only the input element; record coordinates in original pixels.
    with Image.open(source) as image:
        sx, sy = image.width / 1335, image.height / 1888
        coordinates = tuple(round(value * (sx if index % 2 == 0 else sy)) for index, value in enumerate(box))
        crop = image.crop(coordinates)
        crop.thumbnail((1536, 1536))
        target = args.output / f'{task}-crop.png'
        crop.save(target)
    manifest.append({'task': task, 'source_url': url, 'source_sha256': expected_hash, 'crop_coordinates': coordinates, 'crop_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
