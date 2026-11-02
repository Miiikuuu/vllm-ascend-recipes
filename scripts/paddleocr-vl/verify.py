import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

TASKS = {'ocr': 'OCR:', 'table': 'Table Recognition:', 'formula': 'Formula Recognition:', 'chart': 'Chart Recognition:'}


def open_api(request, timeout):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    method = request.get_method() if isinstance(request, urllib.request.Request) else 'GET'
    print(f'API request: {method} {url}', flush=True)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read(1024).decode('utf-8', errors='replace')
        raise OSError(f'API request failed: {method} {url}: HTTP {error.code}; response: {detail}') from error
    except OSError as error:
        raise OSError(f'API request failed: {method} {url}: {error}') from error


def get_json(url):
    with open_api(url, timeout=15) as response:
        return json.load(response)


def check_content(task, content):
    normalized = re.sub(r'\s+', '', content).lower()
    if task == 'ocr':
        expected = ['paddleocr', 'opensource', '2020', 'recognition']
        normalized = re.sub(r'[^a-z0-9]', '', content.lower())
    elif task == 'table':
        expected = ['<fcel>', '<nl>', 'regularhours', 'monday'] + [f'11/{day}/2013' for day in range(1, 15)]
    elif task == 'formula':
        expected = ['f(x)', '2x', 'f(3)', '=27']
        normalized = normalized.replace('\\left', '').replace('\\right', '').replace('{', '').replace('}', '')
    else:
        expected = ['2017', '2024', '37', '22']
    missing = [value for value in expected if value not in normalized]
    return {'expected_fragments': expected, 'missing_fragments': missing, 'otsl_row_count': content.count('<nl>') if task == 'table' else None, 'passed': not missing and (task != 'table' or content.count('<nl>') == 15)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', default='PaddleOCR-VL-0.9B')
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with open_api(args.base_url + '/health', timeout=15) as response:
        assert response.status == 200
    models = get_json(args.base_url + '/v1/models')
    (args.output / 'models.json').write_text(json.dumps(models, indent=2))
    assert any(item['id'] == args.model for item in models['data'])
    results = []
    for repeat in range(-args.warmup, args.repeats):
        for task, prompt in TASKS.items():
            data = (args.samples / (task + '-crop.png')).read_bytes()
            payload = {'model': args.model, 'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}},
                {'type': 'text', 'text': prompt}]}], 'temperature': 0.0, 'max_tokens': 2048, 'seed': 0}
            request = urllib.request.Request(args.base_url + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            start = time.perf_counter()
            with open_api(request, timeout=300) as response:
                result = json.load(response)
            latency = time.perf_counter() - start
            (args.output / f'{task}-{repeat}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
            choice = result['choices'][0]
            check = check_content(task, choice['message']['content'])
            check['finish_reason'] = choice['finish_reason']
            check['passed'] = check['passed'] and choice['finish_reason'] == 'stop'
            row = {'task': task, 'repeat': repeat, 'warmup': repeat < 0, 'input_sha256': hashlib.sha256(data).hexdigest(), 'latency_seconds': latency, 'usage': result.get('usage'), 'validation': check}
            results.append(row)
            (args.output / 'summary.json').write_text(json.dumps({'definition': 'functional sample checks; not a dataset accuracy score', 'concurrency': 1, 'temperature': 0.0, 'seed': 0, 'max_tokens': 2048, 'warmup_rounds': args.warmup, 'repeats': args.repeats, 'results': results}, indent=2, ensure_ascii=False) + '\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)
    if not all(row['validation']['passed'] for row in results):
        raise SystemExit('Functional sample validation failed; inspect raw outputs')


if __name__ == '__main__':
    main()
