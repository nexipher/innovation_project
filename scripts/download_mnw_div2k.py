"""Fetch the approved external dataset only; no inference or training."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'dataset/mnw_div2k'
REV = '42b0188b580bc955d54b26f2ea7016021fc708e7'
GROUPS = ['Adobe_Firefly_v4', 'Google_Imagen4', 'ByteDance_Seedream_v4', 'Midjourney_v7']
ZIP_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
ZIP_BYTES = 448993893


def get(url, **kw):
    r = requests.get(url, timeout=(20, 90), **kw)
    r.raise_for_status()
    return r


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, dest, expected_size, expected_sha=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.stat().st_size == expected_size and (not expected_sha or sha(dest) == expected_sha):
            return
        raise RuntimeError(f'Existing file mismatch; not overwriting: {dest}')
    part = dest.with_name(dest.name + '.part')
    for attempt in range(4):
        try:
            with get(url, stream=True) as r, part.open('wb') as f:
                for block in r.iter_content(1024 * 1024):
                    f.write(block)
            if part.stat().st_size != expected_size:
                raise ValueError(f'Size mismatch: {dest.name}')
            if expected_sha and sha(part) != expected_sha:
                raise ValueError(f'Hash mismatch: {dest.name}')
            part.replace(dest)
            return
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inventory = ROOT / 'download_inventory.json'
    if inventory.exists():
        info = json.loads(inventory.read_text(encoding='utf-8'))
        if info['mnw_revision'] != REV:
            raise RuntimeError('Inventory revision mismatch')
    else:
        tasks = []
        for group in GROUPS:
            url = f'https://github.com/microsoft/MNW/tree/{REV}/AI_Images/{group}'
            html = get(url).text
            names = sorted(set(re.findall(r'AI_Images/'+group+r'/([A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp))', html)))
            if len(names) != 250:
                raise RuntimeError(f'Incomplete listing for {group}: {len(names)}')
            tasks.extend((group, name) for name in names)
        def pointer(task):
            group, name = task
            path = f'AI_Images/{group}/{name}'
            url = f'https://raw.githubusercontent.com/microsoft/MNW/{REV}/{path}'
            for attempt in range(4):
                try:
                    text = get(url).text
                    size = int(re.search(r'^size (\d+)$', text, re.M).group(1))
                    digest = re.search(r'^oid sha256:([a-f0-9]{64})$', text, re.M).group(1)
                    return dict(path=f'Fake/{group}/{name}', group=group, label='Fake',
                                bytes=size, sha256=digest,
                                url=f'https://media.githubusercontent.com/media/microsoft/MNW/{REV}/{path}')
                except (requests.RequestException, AttributeError):
                    if attempt == 3:
                        raise
                    time.sleep(attempt + 1)
        records = []
        with cf.ThreadPoolExecutor(max_workers=16) as pool:
            for i, row in enumerate(pool.map(pointer, tasks), 1):
                records.append(row)
                if i % 200 == 0:
                    print(f'Inventory {i}/1000', flush=True)
        info = dict(mnw_revision=REV, mnw=records, div2k_url=ZIP_URL, div2k_zip_bytes=ZIP_BYTES)
        inventory.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'MNW original bytes: {sum(x["bytes"] for x in info["mnw"]):,}', flush=True)
    def image(row):
        dest = ROOT / row['path']
        fetch(row['url'], dest, row['bytes'], row['sha256'])
        with Image.open(dest) as im:
            im.verify()
        return row
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(image, r) for r in info['mnw']]
        for i, future in enumerate(cf.as_completed(futures), 1):
            future.result()
            if i % 50 == 0:
                print(f'MNW verified {i}/1000', flush=True)
    archive = ROOT / 'downloads/DIV2K_valid_HR.zip'
    print('Downloading DIV2K ZIP (449MB)', flush=True)
    fetch(ZIP_URL, archive, ZIP_BYTES)
    real = []
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        expected = {f'DIV2K_valid_HR/{i:04}.png' for i in range(801, 901)}
        if {m.filename for m in members} != expected:
            raise RuntimeError('Unexpected ZIP members; refusing extraction')
        for member in members:
            data = z.read(member)  # Also verifies CRC.
            dest = ROOT / 'Real' / member.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(data).hexdigest()
            if dest.exists():
                if sha(dest) != digest:
                    raise RuntimeError(f'Existing image mismatch: {dest}')
            else:
                dest.write_bytes(data)
            with Image.open(dest) as im:
                im.verify()
            real.append(dict(path=dest.relative_to(ROOT).as_posix(), label='Real',
                             group='DIV2K_valid_HR', bytes=len(data), sha256=digest, url=ZIP_URL))
    manifest = dict(schema='external_dataset/1', mnw_revision=REV,
                    purpose='External evaluation only; do not use for SFT or threshold fitting',
                    terms={'MNW':'Evaluation only; no training or commercial use',
                           'DIV2K':'Academic research only; original owners retain copyright'},
                    caveats=['Composite benchmark, not an official paired test set',
                             'No guarantee of image-level exclusion from backbone pretraining',
                             'Format, resolution and content confounds require separate analysis'],
                    images_reencoded=False, div2k_zip_sha256=sha(archive),
                    samples=info['mnw'] + sorted(real, key=lambda r:r['path']))
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'COMPLETE: 1000 Fake + {len(real)} Real; original images {sum(r["bytes"] for r in manifest["samples"])/1e9:.3f} GB', flush=True)


if __name__ == '__main__':
    main()
