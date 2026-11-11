"""The formal presentation brand ships with the installed renderer."""
from pathlib import Path
import shutil
import subprocess
import pytest
from PIL import Image
from tarsphere.infra.perception import ego_presentation


def test_packaged_logo_and_real_ffmpeg_composition(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        pytest.skip('ffmpeg unavailable')
    logo = Path(ego_presentation.__file__).resolve().parent / 'assets/tarsphere-logo.png'
    with Image.open(logo) as image:
        assert image.width >= 1921 and image.height >= 644
    for name in ['ego', 'map']:
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                        'color=size=640x480:rate=30', '-t', '0.1', '-y',
                        str(tmp_path / f'{name}.mp4')], check=True)
    ass = tmp_path / 'panel.ass'
    ass.write_text('[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour\nStyle: Default,Arial,20,&H00FFFFFF\n[Events]\nFormat: Layer, Start, End, Style, Text\nDialogue: 0,0:00:00.00,0:00:00.10,Default,Test\n')
    output = tmp_path / 'presentation.mp4'
    ego_presentation.compose_presentation(ego_path=tmp_path/'ego.mp4',
        map_path=tmp_path/'map.mp4', ass_path=ass, output_path=output,
        duration=.1, ffmpeg=ffmpeg)
    assert output.stat().st_size > 0
