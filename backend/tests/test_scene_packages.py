import base64
from copy import deepcopy
import io
import json
import stat
import zipfile
from pathlib import Path

import mujoco
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio import assets, scene_packages as packages, storage
from studio.authoring import SceneProgram, compile_program
from studio.images import save_image, image_path
from studio.main import app
from studio.schemas import Scene
from studio.task_api import get_task

TETRA = 'v -1 -1 -1\nv 1 -1 -1\nv -1 1 -1\nv -1 -1 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path / 'source')


def example(mesh=True, photo=True, robot='cartesian-gripper-v1', legacy=False):
    props = []
    asset = None
    if mesh:
        asset = storage.save('asset', {'name': 'Housing', 'kind': 'mesh', **assets.parse_obj(TETRA)})
        props.append({'name': 'housing', 'shape': 'mesh', 'asset_id': asset['id'], 'position': [-.4, .3, .1], 'size': [.05, .05, .1]})
    program = SceneProgram(name='Portable conveyor', description='A box moves to the tray.', scene=Scene(props=props, embodiment=robot))
    spec, artifact, verification = compile_program(program, None if legacy else get_task('block-to-tray'))
    references = []
    if photo:
        buffer = io.BytesIO()
        Image.new('RGB', (20, 15), 'blue').save(buffer, format='JPEG')
        reference = save_image('Factory.jpg', 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode())
        references.append(reference['id'])
    record = storage.save('scene', {'name': program.name, 'status': 'ready', 'source': 'template', 'builder': 'mujoco',
        'image_ids': references, 'spec': spec, 'artifact_id': storage.write_artifact(artifact), 'verification': verification})
    return record, asset


def unzip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def zipped(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return output.getvalue()


def test_scene_package_roundtrip_into_empty_store_remaps_geometry_photos_and_task(tmp_path, monkeypatch):
    scene, asset = example()
    measurements = [{'label': 'Conveyor length', 'value': 38, 'unit': 'cm'}, {'label': 'Belt width', 'value': 140, 'unit': 'mm'}]
    scene['measurements'] = measurements
    data = packages.package_bytes(scene)
    exported = unzip(data)
    program = json.loads(exported['program.json'])
    program['scene']['cube_y'] = .12
    program['scene']['tray_x'] = .23
    program['waypoints'] = None  # Recompute the teacher for the revised layout.
    exported['program.json'] = json.dumps(program)
    target = tmp_path / 'unrelated' / 'empty-store'
    monkeypatch.setattr(storage, 'ROOT', target)
    imported = packages.import_package(zipped(exported), 'Edited machine')
    assert imported['status'] == 'ready' and imported['builder'] == 'mujoco' and imported['source'] == 'imported'
    assert imported['name'] == 'Edited machine'
    assert imported['measurements'] == measurements
    assert storage.get('scene', imported['id'])['measurements'] == measurements
    assert imported['verification']['status'] == 'pass'
    assert imported['id'] != scene['id'] and imported['asset_ids'] != [asset['id']]
    assert imported['image_ids'] != scene['image_ids']
    assert image_path(imported['image_ids'][0]).is_file()
    assert imported['spec']['scene']['props'][0]['asset_id'] == imported['asset_ids'][0]
    assert imported['spec']['scene']['tray_x'] == .23
    assert storage.get('task', imported['task_id'])['requirements'] == imported['task']['requirements']
    assert imported['task']['requirements'][0]['bounds'][0] == pytest.approx(.23 - .115)
    artifact = storage.read_artifact(imported['artifact_id'])
    model = mujoco.MjModel.from_xml_string(artifact['xml'])
    assert model.geom('prop_housing').type == mujoco.mjtGeom.mjGEOM_MESH
    assert artifact['frames'][-1]['cube'][0] == pytest.approx(.23, abs=.003)
    assert len(assets.list_assets()) == 1
    # Exporting the imported record is itself a supported package.
    again, _ = packages._unpack(packages.package_bytes(imported))
    assert again.name == 'Edited machine'
    assert [measurement.model_dump() for measurement in again.measurements] == measurements


def test_native_export_xml_loads_without_original_absolute_asset_path(tmp_path):
    from studio.adapters.franka import available, asset_folder
    if not available():
        pytest.skip('Official Franka assets are not installed')
    scene, _ = example(mesh=False, photo=False, robot='franka-panda-v1')
    data = packages.package_bytes(scene)
    files = unzip(data)
    assert str(asset_folder().resolve()).encode() not in files['scene.xml']
    assert b'meshdir="robot/assets"' in files['scene.xml']
    assert 'robot/LICENSE' in files and 'robot/panda.xml' in files
    assert b'8161bba264d7fa7c99ca301e91e7fb44737676ad' in files['robot/SOURCE.txt']
    target = tmp_path / 'different-machine'
    for filename, content in files.items():
        path = target / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    model = mujoco.MjModel.from_xml_path(str(target / 'scene.xml'))
    assert model.nq == 16 and model.nmesh > 10


def test_photos_can_be_excluded_and_legacy_scene_still_imports():
    scene, _ = example(mesh=False, legacy=True)
    files = unzip(packages.package_bytes(scene, include_references=False))
    assert json.loads(files['package.json'])['references'] == []
    assert not any(name.startswith('references/') for name in files)
    manifest = json.loads(files['package.json'])
    manifest.pop('measurements')  # Version 1 packages written before measurement preservation.
    files['package.json'] = json.dumps(manifest)
    imported = packages.import_package(zipped(files))
    assert imported['measurements'] == []
    assert imported['task_id'] is None and imported['verification']['status'] == 'pass'


def test_blender_archive_is_preserved_without_loading_or_showing_stale_preview(tmp_path, monkeypatch):
    scene, asset = example(photo=False)
    folder = storage.ROOT / 'blender' / ('a' * 32)
    folder.mkdir(parents=True)
    (folder / 'scene.blend').write_bytes(b'Untrusted archival blend bytes must not execute')
    (folder / 'preview.png').write_bytes(b'original preview')
    source_design = deepcopy(scene['spec'])
    source_design['assemblies'] = [{'name': 'frame', 'parts': [{'position': [-.4,.3,.1], 'size': [.05,.05,.1]}]}]
    scene['builder'] = 'blender'
    scene['blender'] = {'build_id': 'a' * 32, 'program': source_design}
    data = packages.package_bytes(scene)
    monkeypatch.setattr(storage, 'ROOT', tmp_path / 'destination')
    imported = packages.import_package(data)
    assert imported['builder'] == 'mujoco' and not imported.get('blender')
    source = imported['package_source']
    assert source['builder'] == 'blender'
    assert source['blender_program']['scene']['props'][0]['asset_id'] == imported['asset_ids'][0]
    assert source['blender_program']['scene']['props'][0]['asset_id'] != asset['id']
    attachment = packages.source_attachment(source['archive_id'], 'scene.blend')
    assert Path(attachment.path).read_bytes() == b'Untrusted archival blend bytes must not execute'
    with pytest.raises(KeyError):
        packages.source_attachment(source['archive_id'], 'worker.py')
    again = unzip(packages.package_bytes(imported))
    assert again['source/scene.blend'] == b'Untrusted archival blend bytes must not execute'


def test_failed_compile_rolls_back_new_assets_files_and_records(monkeypatch):
    scene, _ = example()
    raw = packages.package_bytes(scene)
    before = {kind: storage.all_records(kind) for kind in ('asset', 'scene', 'task', 'reference')}
    def fail(*args):
        assert len(assets.list_assets()) == 1  # Staged import meshes remain hidden.
        raise ValueError('Unusable imported geometry')
    monkeypatch.setattr(packages, 'compile_program', fail)
    with pytest.raises(ValueError):
        packages.import_package(raw)
    for kind, records in before.items():
        assert storage.all_records(kind) == records
    assert not list((storage.ROOT / 'packages').glob('*'))


def test_atomic_publication_rolls_back_artifact_and_photo_files(monkeypatch):
    scene, _ = example()
    raw = packages.package_bytes(scene)
    before_artifacts = set((storage.ROOT / 'artifacts').iterdir())
    before_photos = set((storage.ROOT / 'references').iterdir())
    original = storage.write_artifact
    def fail_after_write(data):
        raise OSError('Disk failure')
    monkeypatch.setattr(storage, 'write_artifact', fail_after_write)
    with pytest.raises(OSError): packages.import_package(raw)
    assert set((storage.ROOT / 'artifacts').iterdir()) == before_artifacts
    assert set((storage.ROOT / 'references').iterdir()) == before_photos
    assert len(storage.all_records('asset')) == 1
    assert len(storage.all_records('scene')) == 1


@pytest.mark.parametrize('filename', ['../outside.py', '/tmp/outside', 'assets/../outside', 'assets\\outside', 'worker.py'])
def test_archive_rejects_unsafe_or_unlisted_files(filename):
    scene, _ = example(mesh=False, photo=False)
    files = unzip(packages.package_bytes(scene)); files[filename] = b'no'
    with pytest.raises(ValueError): packages.import_package(zipped(files))


def test_archive_rejects_symlink_duplicate_and_decompression_limit(monkeypatch):
    scene, _ = example(mesh=False, photo=False)
    raw = packages.package_bytes(scene)
    output = io.BytesIO(raw)
    with zipfile.ZipFile(output, 'a') as archive:
        link = zipfile.ZipInfo('link'); link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, '../outside')
    with pytest.raises(ValueError): packages.import_package(output.getvalue())
    output = io.BytesIO(raw)
    with zipfile.ZipFile(output, 'a') as archive:
        with pytest.warns(UserWarning): archive.writestr('package.json', '{}')
    with pytest.raises(ValueError): packages.import_package(output.getvalue())
    monkeypatch.setattr(packages, 'MAX_EXPANDED_BYTES', 1024)
    with pytest.raises(ValueError): packages.import_package(raw)


@pytest.mark.parametrize('mutation', [
    lambda f: f.update({'package.json': json.dumps({'format': 'other', 'version': 1})}),
    lambda f: f.update({'program.json': json.dumps({**json.loads(f['program.json']), 'python': 'do not run'})}),
    lambda f: f.update({'task.json': json.dumps({**json.loads(f['task.json']), 'requirements': [{'check': 'execute_python'}]})}),
    lambda f: f.update({'program.json': f['program.json'].replace(b'"seed": 42', b'"seed": NaN')}),
])
def test_import_validates_declarative_program_format_and_task(mutation):
    scene, _ = example(mesh=False, photo=False)
    files = unzip(packages.package_bytes(scene)); mutation(files)
    with pytest.raises(ValueError): packages.import_package(zipped(files))
    assert len(storage.all_records('scene')) == 1


def test_package_api_download_import_and_sanitized_errors():
    scene, _ = example()
    with TestClient(app) as client:
        response = client.get(f'/api/scenes/{scene["id"]}/package?include_references=false')
        assert response.status_code == 200 and response.headers['content-type'] == 'application/zip'
        assert response.headers['x-content-type-options'] == 'nosniff'
        uploaded = client.post('/api/scenes/import', json={'name': 'Imported API scene', 'data_base64': base64.b64encode(response.content).decode()})
        assert uploaded.status_code == 201, uploaded.text
        assert uploaded.json()['name'] == 'Imported API scene'
        assert uploaded.json()['image_ids'] == []
        invalid = client.post('/api/scenes/import', json={'data_base64': 'PRIVATE_INVALID_ZIP'})
        assert invalid.status_code == 422 and 'PRIVATE_INVALID_ZIP' not in invalid.text


def test_repacked_archive_allows_declared_directories_and_rebinds_legacy_tray():
    scene, _ = example(legacy=True)
    files = unzip(packages.package_bytes(scene))
    program = json.loads(files['program.json'])
    program['scene']['tray_x'] = .2
    program['waypoints'] = None
    files['program.json'] = json.dumps(program)
    output = io.BytesIO(zipped(files))
    with zipfile.ZipFile(output, 'a') as archive:
        archive.writestr('assets/', b'')
        archive.writestr('references/', b'')
    imported = packages.import_package(output.getvalue())
    assert imported['spec']['scene']['tray_x'] == .2
    assert imported['verification']['status'] == 'pass'


def test_native_import_requires_installed_runtime_without_installing_archive_code(monkeypatch):
    from studio.adapters import franka
    scene, _ = example(mesh=False, photo=False)
    files = unzip(packages.package_bytes(scene))
    program = json.loads(files['program.json'])
    program['scene']['embodiment'] = 'franka-panda-v1'
    files['program.json'] = json.dumps(program)
    monkeypatch.setattr(franka, 'available', lambda: False)
    with TestClient(app) as client:
        response = client.post('/api/scenes/import', json={'data_base64': base64.b64encode(zipped(files)).decode()})
        assert response.status_code == 409
        assert 'Install the supplied Franka' in response.json()['detail']


@pytest.mark.parametrize('measurements', [
    [{'label': 'Length', 'value': -1, 'unit': 'm'}],
    [{'label': 'Length', 'value': 4, 'unit': 'feet'}],
    [{'label': 'Length', 'value': 1, 'unit': 'm'}] * 9,
])
def test_package_rejects_out_of_bounds_measurements(measurements):
    scene, _ = example(mesh=False, photo=False)
    files = unzip(packages.package_bytes(scene))
    manifest = json.loads(files['package.json'])
    manifest['measurements'] = measurements
    files['package.json'] = json.dumps(manifest)
    with pytest.raises(ValueError): packages.import_package(zipped(files))
    assert len(storage.all_records('scene')) == 1
