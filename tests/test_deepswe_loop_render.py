import json
import pytest
pytest.importorskip('PIL')
from scripts.render_deepswe_loop_gif import MemorySeries,loop_position,visible_lines


def test_vram_displays_only_preceding_actual_sample(tmp_path):
    d=tmp_path/'native';d.mkdir()
    (d/'phase-times.json').write_text(json.dumps(dict(resume_begin=100)))
    rows=[dict(at=99.8,memory_used_bytes=100,memory_total_bytes=1000,memory_free_bytes=850,memory_reserved_bytes=50),
          dict(at=100.7,memory_used_bytes=200,memory_total_bytes=1000,memory_free_bytes=750,memory_reserved_bytes=50)]
    (d/'gpu-telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    series=MemorySeries(tmp_path,'native')
    assert series.at(0)[0]['memory_used_bytes']==100
    assert series.at(.69)[0]['memory_used_bytes']==100
    assert series.at(.71)[0]['memory_used_bytes']==200
    assert series.at(.71)[1]==pytest.approx(.01)


def test_backends_repeat_complete_capture_at_same_elapsed_speed():
    assert loop_position(12.5,10)==(1,2.5)
    assert loop_position(12.5,3)==(4,.5)
    assert loop_position(20,10)==(2,0)
    with pytest.raises(ValueError):loop_position(0,0)


def test_english_blank_line_marker_preserves_original_text():
    assert visible_lines('original\n\n\ntext')==['original','[2 blank lines]','text']
    assert visible_lines('原始模型输出')==['原始模型输出']
