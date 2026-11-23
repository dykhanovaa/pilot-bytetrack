"""Удобный запуск технического пилота без изменения исходных результатов.

.venv/bin/python run_pilot.py --mode synthetic
.venv/bin/python run_pilot.py --mode real
.venv/bin/python run_pilot.py --mode all
"""
from pathlib import Path
from datetime import datetime
import argparse
import contextlib
import csv
import importlib.metadata
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def check_environment():
    for name, required in [('ultralytics', '8.3.0'), ('numpy', '1.26.4'), ('torch', '2.4.1')]:
        actual = importlib.metadata.version(name).split('+')[0]
        if actual != required:
            raise RuntimeError(f'{name}: требуется {required}, установлено {actual}. Используйте .venv/bin/python.')


def summary(folder):
    source = folder / 'synthetic_summary.json'
    if not source.exists():
        return
    rows = json.loads(source.read_text())
    rows = [r for r in rows if r['scenario'] == 'lifecycle']
    with (folder / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('\nПоздний объект B: выданные ID до первого исчезновения')
    print(' k | штатный ByteTrack | адаптация SkipAware')
    for k in range(1, 11):
        a = next(r for r in rows if r['mode'] == 'stock' and r['k'] == k)
        b = next(r for r in rows if r['mode'] == 'skipaware' and r['k'] == k)
        print(f'{k:2} | {str(a["late_B"]):17} | {b["late_B"]}')
    # One row per original frame; full JSON preserves coordinates and states too.
    with (folder / 'states.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario','mode','k','frame','observed','detections_xyxy_conf_cls',
                         'unconfirmed_ids','tracked_ids','lost_ids','removed_ids','output_ids'])
        for scenario in ['lifecycle', 'empty_observation']:
            for mode in ['stock','skipaware']:
                for k in range(1, 11):
                    log = json.loads((folder / f'synthetic_{scenario}_{mode}_k{k}.json').read_text())
                    for row in log:
                        tracks = row['tracks']
                        def ids(state, activated=None):
                            return [t['id'] for t in tracks if t['state'] == state
                                    and (activated is None or t['confirmed'] == activated)]
                        writer.writerow([scenario,mode,k,row['frame'],row['observed'],
                            json.dumps(row['input_xyxy_conf_cls']), ids('tracked',False),
                            ids('tracked',True),ids('lost'),ids('removed'),
                            [int(x[4]) for x in row['output']]])


def real_prerequisites():
    sources = ROOT / 'data' / 'sources.json'
    if not sources.exists():
        raise RuntimeError('Нет данных. Сначала запустите: .venv/bin/python fetch_data.py')
    paths = [ROOT / x['file'] for x in json.loads(sources.read_text())['files']]
    paths += [ROOT/'yolov8n.pt',ROOT/'vendor'/'TrackEval'/'trackeval'/'__init__.py']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError('Не хватает файлов: ' + ', '.join(missing[:3]) + '. См. QUICKSTART.txt.')


def main():
    parser = argparse.ArgumentParser(description='Технический пилот ByteTrack: k от 1 до 10')
    parser.add_argument('--mode', choices=['synthetic','real','all'], default='synthetic')
    args = parser.parse_args()
    folder = ROOT / 'runs' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    folder.mkdir(parents=True)
    print('Результаты:', folder, flush=True)
    try:
        check_environment()
        if args.mode in ['synthetic','all']:
            import synthetic
            synthetic.OUT = folder
            with (folder/'synthetic.log').open('w') as log, contextlib.redirect_stdout(log):
                synthetic.main()
            summary(folder)
            print('Искусственные сценарии: PASS. Журнал: synthetic.log', flush=True)
        if args.mode in ['real','all']:
            real_prerequisites()
            import patch_trackeval  # Idempotent NumPy alias patch, no metric changes.
            import pilot
            pilot.OUT = folder
            print('Реальный пилот: 3 фрагмента × 120 кадров. Дождитесь завершения.', flush=True)
            # No cached detections from earlier evidence are loaded into a new run folder.
            with (folder/'real.log').open('w') as log, contextlib.redirect_stdout(log):
                pilot.main()
            print('Реальный пилот: PASS. Метрики: metrics_full_frames.csv; кадры: frames/', flush=True)
        print('Готово:', folder)
    except Exception:
        (folder/'error.log').write_text(traceback.format_exc())
        print('Ошибка. Подробности:', folder/'error.log', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
