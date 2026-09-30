"""Render a clearly labeled synthetic-teacher/learned-policy GIF, from actual states."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ghosthands.simulation import Tabletop, HORIZON
from ghosthands.data import teacher
from ghosthands.models import LearnedAgent


def font(size: int):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rollout(env, action):
    states = [env.state.copy()]
    for _ in range(HORIZON):
        env.step(action(env))
        states.append(env.state.copy())
        if env.success:
            break
    return states, env.metrics()


def panel(draw, state, trail, left):
    def p(x):
        return (int(left+220+(x[0]-x[1])*200), int(185+(x[0]+x[1])*70-x[2]*200))
    draw.polygon([p(v) for v in [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]], fill="#172631", outline="#405665")
    for k in np.linspace(0,1,11):
        draw.line([p([k,0,0]),p([k,1,0])],fill="#2b3c48")
        draw.line([p([0,k,0]),p([1,k,0])],fill="#2b3c48")
    if len(trail)>1:
        draw.line([p(s[:3]) for s in trail],fill="#719d9b",width=2)
    tx,ty=p(state[8:11]); draw.ellipse([tx-17,ty-8,tx+17,ty+8],outline="#87f0c7",width=2)
    ox,oy=p(state[4:7]);draw.polygon([(ox-12,oy-5),(ox+5,oy-12),(ox+15,oy-5),(ox+15,oy+10),(ox-3,oy+17),(ox-12,oy+9)],fill="#e98a89",outline="#ffbeb0")
    ex,ey=p(state[:3]);gap=8 if state[12] else 17
    draw.line([(ex-gap,ey+3),(ex-gap,ey-20),(ex+gap,ey-20),(ex+gap,ey+3)],fill="#94f5d4",width=4)
    draw.line([(ex,ey),(ex,int(185+(state[0]+state[1])*70))],fill="#526975",width=1)


def main():
    import torch
    torch.set_num_threads(2)
    agent=LearnedAgent(Path("assets/relative.pt"))
    a,am=rollout(Tabletop(0),teacher)
    b,bm=rollout(Tabletop(20000,True),lambda e:agent.action(e.state))
    frames=[]
    for t in range(max(len(a),len(b))+22):
        im=Image.new("RGB",(960,470),"#0a0e13");d=ImageDraw.Draw(im)
        d.text((35,22),"GhostHands",font=font(30),fill="#e9eff5")
        d.text((35,63),"Learning a skill. Transferring it to a new layout.",font=font(16),fill="#a3b2c4")
        d.text((35,114),"01  SYNTHETIC TEACHER",font=font(13),fill="#adbae8")
        d.text((515,114),"02  LEARNED RELATIVE BC",font=font(13),fill="#8af3ce")
        panel(d,a[min(t,len(a)-1)],a[:t+1],20)
        panel(d,b[min(t,len(b)-1)],b[:t+1],500)
        d.text((35,359),"Training example / seed 0",font=font(14),fill="#9aafc1")
        status="Placed" if t>=len(b)-1 and bm["success"] else "Executing" if t<len(b)-1 else "Failed"
        d.text((515,359),f"New layout / seed 20000 / {status}",font=font(14),fill="#9aafc1")
        d.line((35,398,925,398),fill="#263b47")
        d.text((35,417),"240 synthetic training episodes · neural actions · kinematic simulation",font=font(14),fill="#8798aa")
        frames.append(im)
    frames[0].save("assets/demo.gif",save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    print({"teacher":am,"learned":bm,"frames":len(frames)})


if __name__=="__main__":
    main()
