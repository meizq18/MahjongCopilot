import ast
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from visual_state.core import Collector, UnknownClassifier, normalize, mjai_tile, hand_boxes, MjaiBridge, publish_observation

def image(n=13):
    a=np.zeros((1080,1920,3),np.uint8)
    for i in range(n):
        x=235+i*100
        a[860:1012,x:x+95]=240
    return a

@pytest.mark.parametrize('tile,expected',[('0m','5mr'),('0p','5pr'),('0s','5sr'),('1z','E'),('7z','C'),('9s','9s')])
def test_mjai(tile,expected): assert mjai_tile(tile)==expected

def test_invalid_tile():
    with pytest.raises(ValueError): mjai_tile('back')

@pytest.mark.parametrize('n',[1,7,13,14])
def test_segmentation(n): assert len(hand_boxes(image(n)))==n

def test_blank(): assert hand_boxes(image(0))==[]

def test_bounds():
    with pytest.raises(ValueError): normalize(image(),(-1,0,1920,1080))
    with pytest.raises(ValueError): normalize(image(),(0,0,100,100))

def test_scaled_rectangle():
    canvas=np.zeros((600,1100,3),np.uint8)
    canvas[20:560,30:990]=cv2.resize(image(),(960,540))
    assert len(hand_boxes(normalize(canvas,(30,20,960,540))))==13

def test_stable_and_invalidate():
    c=Collector(lambda _: '1m')
    a=image(3)
    assert not c.observe(a,(0,0,1920,1080),0).stable
    assert not c.observe(a,(0,0,1920,1080),.2).stable
    s=c.observe(a,(0,0,1920,1080),.4)
    assert s.stable and not s.model_ready and not s.history_complete
    assert s.self_seat.value is None and s.players[1].river.value is None
    assert not c.observe(image(0),(0,0,1920,1080),.6).stable
    assert not c.observe(a,(0,0,1920,1080),.8).stable

def test_unknown():
    c=Collector(UnknownClassifier())
    for t in (0,.2,.4):
        s=c.observe(image(),(0,0,1920,1080),t)
        assert not s.stable and 'unrecognized_tile' in s.issues

def test_red_five_multiplicity():
    class C:
        i=0
        def __call__(self,_):
            self.i+=1
            return '0m' if self.i==1 else '5m'
    assert 'impossible_tile_multiplicity' in Collector(C()).observe(image(5),(0,0,1920,1080),0).issues

def test_gap_and_timestamp():
    c=Collector(lambda _: '1m',stable_frames=2)
    c.observe(image(3),(0,0,1920,1080),0)
    assert c.observe(image(3),(0,0,1920,1080),.2).stable
    s=c.observe(image(3),(0,0,1920,1080),2)
    assert not s.stable and 'capture_gap_history_unknown' in s.issues
    with pytest.raises(ValueError): c.observe(image(3),(0,0,1920,1080),2)

def test_bridge_fails_closed():
    s=Collector(lambda _: '1m').observe(image(3),(0,0,1920,1080),0)
    with pytest.raises(RuntimeError): MjaiBridge().events(s)

def test_hud_contract():
    class Hud:
        def __init__(self): self.calls=[]
        def overlay_clear_guidance(self): self.calls.append('clear')
        def overlay_update_guidance(self,*args): self.calls.append(args)
    h=Hud()
    publish_observation(h,Collector(UnknownClassifier()).observe(image(),(0,0,1920,1080),0))
    assert h.calls[0]=='clear' and h.calls[1][2]==[]

def test_cli(tmp_path):
    cv2.imwrite(str(tmp_path/'001.png'),image())
    output=tmp_path/'output.jsonl'
    subprocess.run([sys.executable,'-m','visual_state','--frames',str(tmp_path),'--rect','0','0','1920','1080','--output',str(output)],check=True)
    s=json.loads(output.read_text())
    assert len(s['hand']['value'])==13 and not s['model_ready']
    assert s['timestamp_source']=='synthetic'

def test_no_forbidden_imports():
    forbidden={'pyautogui','mitmproxy','liqi','sdk','bot_manager','socket','requests','selenium'}
    for path in Path(__file__).parents[1].glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import): assert not any(n.name.split('.')[0] in forbidden for n in node.names)
            elif isinstance(node,ast.ImportFrom): assert (node.module or '').split('.')[0] not in forbidden

def test_classifier_module_loaded_without_gui(tmp_path):
    from visual_state.core import WrapperClassifier
    (tmp_path/'action').mkdir()
    (tmp_path/'__init__.py').write_text("raise RuntimeError('must not import package')")
    (tmp_path/'action'/'action.py').write_text("raise RuntimeError('must not import GUI')")
    (tmp_path/'action'/'classifier.py').write_text("class Classify:\n def __call__(self, image): return '0p'\n")
    assert WrapperClassifier(tmp_path)(image())=='0p'
