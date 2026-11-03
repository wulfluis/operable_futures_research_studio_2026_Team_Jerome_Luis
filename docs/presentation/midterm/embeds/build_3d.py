"""Builds the 3D wireframe versions of the Surfside and Rana Plaza animations (deck slides 2 and 3).
Same style as the 2D ones: black, white lines, red cracks with detection boxes, captions, REC, step counter."""
from pathlib import Path

OUT = Path(__file__).parent  # writes surfside.html and ranaplaza.html; the deck holds them as surfside_3d / ranaplaza_3d

HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0b0b0b;overflow:hidden;font-family:Consolas,"Courier New",monospace;color:#f2f2f2}
canvas{position:absolute;inset:0;width:100%;height:100%}
#cap{position:absolute;left:28px;right:28px;bottom:22px}
#t{font-size:26px;font-weight:700;letter-spacing:1px;text-transform:uppercase}
#u{font-size:20px;color:#bfbfbf;margin-top:6px;min-height:24px}
#step{position:absolute;left:28px;top:22px;font-size:20px;color:#a6a6a6}
#rec{position:absolute;right:28px;top:22px;font-size:20px;color:#d62728}
#hud{position:absolute;right:28px;top:50px;font-size:15px;color:#6f6f6f;text-align:right}
#bar{position:absolute;left:0;bottom:0;height:3px;background:#d62728;width:0}
#end{position:absolute;left:0;right:0;top:150px;text-align:center;font-size:44px;font-weight:700;text-transform:uppercase;opacity:0;transition:opacity 1s}
</style></head><body>
<canvas id="c"></canvas>
<div id="step">01 / 06</div><div id="rec">&#9679; REC</div><div id="hud"></div>
<div id="end">%END%</div>
<div id="cap"><div id="t"></div><div id="u"></div></div>
<div id="bar"></div>
<script>
"""

# a tiny wireframe engine: orbiting perspective camera, 3D lines, transforms for falling parts, CV boxes
ENGINE = r"""
var c=document.getElementById("c"),g=c.getContext("2d"),W,H,DP=window.devicePixelRatio||1;
function rs(){W=c.clientWidth;H=c.clientHeight;c.width=W*DP;c.height=H*DP;g.setTransform(DP,0,0,DP,0,0)}rs();window.onresize=rs;
function P(p){var X=p[0]-cam.tx,Y=p[1]-cam.ty,Z=p[2]-cam.tz,a=Math.cos(cam.yaw),b=Math.sin(cam.yaw),
x1=X*a-Z*b,z1=X*b+Z*a,cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch),y2=Y*cp+z1*sp,z2=-Y*sp+z1*cp+cam.dist;
return[W/2+x1*cam.f/z2,H*cam.cy-y2*cam.f/z2]}
function T(p,t){if(!t)return p;var x=p[0]-t.px,y=p[1]-t.py,a=Math.cos(t.r||0),b=Math.sin(t.r||0);
return[t.px+x*a-y*b+(t.dx||0),t.py+x*b+y*a+(t.dy||0),p[2]]}
function L(ps,col,w,al,t,dash){g.beginPath();for(var i=0;i<ps.length;i++){var q=P(T(ps[i],t));i?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1])}
g.strokeStyle=col;g.lineWidth=w;g.globalAlpha=al;g.setLineDash(dash||[]);g.stroke();g.globalAlpha=1}
function B(x0,y0,z0,x1,y1,z1,col,w,al,t){[[[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1],[x0,y0,z0]],
[[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1],[x0,y1,z0]]].forEach(function(r){L(r,col,w,al,t)});
[[x0,z0],[x1,z0],[x1,z1],[x0,z1]].forEach(function(e){L([[e[0],y0,e[1]],[e[0],y1,e[1]]],col,w,al,t)})}
function grid(y,x0,x1,z0,z1,s){for(var x=x0;x<=x1;x+=s)L([[x,y,z0],[x,y,z1]],"#2c2c2c",1,1);for(var z=z0;z<=z1;z+=s)L([[x0,y,z],[x1,y,z]],"#2c2c2c",1,1)}
function zig(x,y0,y1,z,n,amp,seed){var r=[],s=seed;for(var i=0;i<=n;i++){s=(s*9301+49297)%233280;r.push([x+(i%2?amp:-amp)*(.5+s/233280),y0+(y1-y0)*i/n,z])}return r}
function box(ps,lab,al,t){var a=1e9,b=1e9,cc=-1e9,d=-1e9;ps.forEach(function(p){var q=P(T(p,t));a=Math.min(a,q[0]);b=Math.min(b,q[1]);cc=Math.max(cc,q[0]);d=Math.max(d,q[1])});
a-=7;b-=7;cc+=7;d+=7;g.globalAlpha=al;g.setLineDash([]);g.strokeStyle="#d62728";g.lineWidth=1.5;g.strokeRect(a,b,cc-a,d-b);
if(lab){g.font="bold 13px Consolas,monospace";var w=g.measureText(lab).width+8;g.fillStyle="#d62728";g.fillRect(a,b-17,w,17);g.fillStyle="#0b0b0b";g.fillText(lab,a+4,b-4)}g.globalAlpha=1}
function txt(p,s,col,al){var q=P(p);g.globalAlpha=al;g.font="13px Consolas,monospace";g.fillStyle=col;g.fillText(s,q[0],q[1]);g.globalAlpha=1}
function corners(){g.strokeStyle="#d62728";g.lineWidth=3;g.setLineDash([]);g.beginPath();var m=12,k=40;
[[m,m,1,1],[W-m,m,-1,1],[m,H-m,1,-1],[W-m,H-m,-1,-1]].forEach(function(e){g.moveTo(e[0],e[1]+e[3]*k);g.lineTo(e[0],e[1]);g.lineTo(e[0]+e[2]*k,e[1])});g.stroke()}
function cl(v){return v<0?0:v>1?1:v}
function rnd(n){var x=Math.sin(n*127.1)*43758.5;return x-Math.floor(x)}
function rubble(n,x0,x1,y0,y1,z0,z1,gy,sp,k,al,sd){for(var j=0;j<n;j++){var a=rnd(sd+j),b=rnd(sd+j+.3),c=rnd(sd+j+.7),d=rnd(sd+j+.9),
kj=cl((k-.15*a)/(.85)),sx=x0+(x1-x0)*a,sy=y0+(y1-y0)*b,sz=z0+(z1-z0)*c,ex=sx+(d-.5)*sp,ez=sz+(rnd(sd+j+.5)-.5)*sp,
z=.12+.3*d,cx=sx+(ex-sx)*kj,cz=sz+(ez-sz)*kj,cy=sy+(gy+z-sy)*kj*kj;if(k<=0)continue;
B(cx-z,cy-z,cz-z*.7,cx+z,cy+z*.6,cz+z*.7,j%3?"#bfbfbf":"#f2f2f2",1,al*Math.min(1,k*4),{px:cx,py:cy,r:kj*(a*6-3)})}}
function smoke(x0,x1,z0,z1,gy,tm,al,sd){if(tm<=0)return;for(var j=0;j<34;j++){var a=rnd(sd+j),b=rnd(sd+j+.4),c=rnd(sd+j+.8),
age=tm-a*.5;if(age<=0)continue;var x=x0+(x1-x0)*b+(b-.5)*age*1.6,z=z0+(z1-z0)*c+(c-.5)*age*1.6,y=gy+.3+age*(.35+.5*a),
r=.6+age*(.9+a),q=P([x,y,z]),q2=P([x+r,y,z]),pr=Math.abs(q2[0]-q[0])+4,o=al*.22*Math.min(1,age*3)*Math.max(0,1-age/6),
gr=g.createRadialGradient(q[0],q[1],0,q[0],q[1],pr);gr.addColorStop(0,"rgba(150,150,150,"+o+")");gr.addColorStop(1,"rgba(150,150,150,0)");
g.fillStyle=gr;g.beginPath();g.arc(q[0],q[1],pr,0,6.3);g.fill()}}
var Tt=0;S.forEach(function(s){Tt+=s[0]});
var te=document.getElementById("t"),ue=document.getElementById("u"),st=document.getElementById("step"),
bar=document.getElementById("bar"),en=document.getElementById("end"),hud=document.getElementById("hud"),
t0=Date.now()-(parseFloat(location.hash.slice(1))||0)*700,cur=-1;
function f(){var ms=Date.now()-t0,e=(ms/700)%Tt,a=0,i=0;while(e>=a+S[i][0]){a+=S[i][0];i++}
if(i!=cur){cur=i;te.innerHTML=S[i][1];ue.innerHTML=S[i][2];st.textContent="0"+(i+1)+" / 06";en.style.opacity=i==5?1:0}
cam.yaw=YAW+.32*Math.sin(ms/5200);g.clearRect(0,0,W,H);
hud.innerHTML="CAM 01 &middot; YAW "+Math.round(cam.yaw*57.3)+"&deg;<br>WIREFRAME &middot; NOT TO SCALE";
scene(i,e-a,ms/1000);corners();bar.style.width=(e/Tt*100)+"%";requestAnimationFrame(f)}f();
"""

SURFSIDE = r"""
var S=[[4,"1981 &middot; Champlain Towers South","Residential tower; garage under a flat pool deck"],
[5,"Water on a flat deck","Failed waterproofing lets water into the slab below (2018 report)"],
[5,"2018 &middot; Cracks documented","Abundant cracking and spalling in garage columns, beams and walls"],
[4,"Three years pass","The cracks are already in the report&rsquo;s photos"],
[5,"24 Jun 2021 &middot; Partial collapse","98 dead &middot; cause under NIST investigation"],
[4,"",""]],YAW=-.62,cam={pitch:.36,dist:25,tx:7,ty:2.4,tz:2.5,f:640,cy:.45};
var CR=[zig(7,-.15,-1.05,0,5,.12,3),zig(9,-.2,-1.0,0,4,.1,7),zig(11,-.1,-1.05,0,5,.12,11),zig(13,-.2,-.95,0,4,.1,5)];
function scene(i,l,s){var dim=i==5?.25:1,k=i==4?cl(l/1.5):i==5?1:0;k=k*k;
grid(-1.2,-2,16,-3,8,2);
var deck={px:10,py:0,dy:-1.05*k,r:.05*k},east={px:4.5,py:0,dy:-7.5*k,r:.14*k};
L([[0,-1.2,0],[14,-1.2,0],[14,-1.2,5],[0,-1.2,5],[0,-1.2,0]],"#6f6f6f",1,dim);
for(var x=0;x<=14;x+=2)for(var z=0;z<=5;z+=2.5)L([[x,-1.2,z],[x,0,z]],x>5?"#bfbfbf":"#6f6f6f",x>5?2:1,dim*(x>5&&k>.3?.25:1),x>5?deck:null);
B(0,0,0,3,9,5,"#f2f2f2",1.6,dim);for(var y=.75;y<9;y+=.75)L([[0,y,0],[3,y,0],[3,y,5]],"#6f6f6f",1,dim);
var K=i==4?cl(l/1.5):i==5?1:0;[[3,0],[6,0],[6,5],[3,5]].forEach(function(e){L([[e[0],0,e[1]],[e[0],9,e[1]]],"#f2f2f2",1.6,dim*(1-K))});
for(var j=0;j<=12;j++){var kj=cl((K*1.6-(12-j)*.06)/1.1);kj*=kj;var y0=j*.75,tj={px:4.5,py:y0,dy:(-1.2+j*.09-y0)*kj,r:(j%2?1:-1)*.06*kj};
L([[3,y0,0],[6,y0,0],[6,y0,5],[3,y0,5],[3,y0,0]],j%12?"#6f6f6f":"#f2f2f2",j%12?1:1.6,dim*(i==5?.6:1),tj)}
if(i<5){B(6,-.15,0,14,0,5,"#f2f2f2",2.4,(1-.75*k)*dim,deck);B(8.5,0,1.2,12.5,.35,3.8,"#8c8c8c",1,(1-.75*k)*dim,deck)}
if(i>=1&&i<=3)for(var d=0;d<14;d++){var dx=7+((d*37)%70)/10,dz=.5+((d*53)%40)/10,p=((s*.7+d*.29)%1);
L([[dx,-.05-p*1.1,dz],[dx,-.2-p*1.1,dz]],"#8c8c8c",2,1-p)}
if(i>=2&&i<=4)CR.forEach(function(r){L(r,"#d62728",2.6,dim*(1-.8*k),deck)});
if(i>=2&&i<=3)CR.forEach(function(r,j){box(r,j?"":"CRACK 0.94",1,deck)});
if(i>=4){var tc=i==4?l:l+5;rubble(26,3,6,1,9,0,5,-1.2,5,K,dim*(i==5?1.6:1),11);rubble(14,6.5,13.5,-.2,0,0,5,-1.2,1.6,k,dim*(i==5?1.6:1),41);
smoke(2.5,8,0,5,-1.2,tc*.7-.5,i==5?.6:1,7)}
if(i>=4){var r=[],q=1;for(x=3;x<=14;x+=.5){q=(q*9301+49297)%233280;r.push([x,-1.2+.15+.45*q/233280,2.5])}L(r,"#f2f2f2",1.6,k*(i==5?.4:1))}
}
"""

RANA = r"""
var S=[[4,"Savar, Dhaka &middot; Rana Plaza","Eight storeys: shops and a bank below, garment factories above"],
[5,"23 Apr 2013 &middot; Cracks appear","A local TV crew films them"],
[3,"The building is evacuated",""],
[5,"The owner says it is safe","Workers are told to come back the next day"],
[5,"24 Apr 2013 &middot; Morning collapse","1,134 dead &middot; about 2,500 injured"],
[4,"",""]],YAW=-.5,cam={pitch:.34,dist:24,tx:4.5,ty:3.6,tz:2.5,f:640,cy:.46};
var CR=[zig(3,.08,.92,0,4,.1,2),zig(4.5,.1,.9,0,4,.1,9),zig(6,.1,.92,0,4,.1,4),
[[6.9,1.3,0],[7.3,1.5,0],[7.7,1.42,0],[8.1,1.65,0],[8.5,1.58,0]]];
function scene(i,l,s){var dim=i==5?.2:1;grid(0,-6,14,-6,8,2);
var wk=i==2?1-cl(l/1):i==3?cl(l/1):i>=4?0:1;
for(var f=0;f<8;f++){var kk=i==4?cl((l-(7-f)*.09)/1.3):i==5?1:0;kk*=kk;
var t={px:4.5,py:f,dy:-f*.86*kk,r:(f%2?1:-1)*.035*kk},al=dim*(i==5?1:1-.75*kk);
B(0,f,0,9,f+1,5,"#f2f2f2",1.4,al,t);for(var x=1.5;x<9;x+=1.5)L([[x,f,0],[x,f+1,0]],"#6f6f6f",1,al,t);
if(f>=2&&i<4)for(var w=0;w<14;w++)L([[.4+w*.6,f+.25,.6],[.4+w*.6,f+.42,.6]],"#f2f2f2",3,wk*dim,t);
if(f<2&&i<4)txt(T([.3,f+.35,0],t),f?"BANK":"SHOPS","#8c8c8c",dim)}
if(i>=1&&i<=3)CR.forEach(function(r){L(r,"#d62728",2.6,1)});
if(i>=1&&i<=3){box(CR[0].concat(CR[1],CR[2]),"CRACK 0.93",1);box(CR[3],"CRACK 0.87",1)}
if(i==1){var cx=-3.2,cy=1,cz=-5;B(cx-.5,cy-.3,cz-.4,cx+.5,cy+.3,cz+.4,"#f2f2f2",1.4,1);
L([[cx+.5,cy,cz+.4],[3,.5,0]],"#8c8c8c",1,1,null,[5,5]);L([[cx+.5,cy,cz+.4],[6,.5,0]],"#8c8c8c",1,1,null,[5,5]);
txt([cx-.6,cy-1.4,cz],"LOCAL TV","#a6a6a6",1)}
if(i>=4){var tc=i==4?l:l+5;rubble(34,0,9,1,8,0,5,0,4,cl(l/2)+(i==5?1:0),dim*(i==5?2:1),23);smoke(-.5,9.5,0,5,0,tc*.7-.6,i==5?.6:1,3)}
if(i>=4){var k=cl(l/1.6),r=[],q=3;for(var x2=-.5;x2<=9.5;x2+=.4){q=(q*9301+49297)%233280;r.push([x2,.2+1.1*q/233280,2.5])}L(r,"#f2f2f2",1.6,k*(i==5?.4:1))}
}
"""

for name, scene, end in [("surfside", SURFSIDE, 'Documented in 2018.<br><span style="color:#d62728">Collapsed in 2021.</span>'),
                         ("ranaplaza", RANA, 'The cracks were seen<br><span style="color:#d62728">the day before.</span>')]:
    page = HEAD.replace("%END%", end) + scene + ENGINE + "</script></body></html>"
    (OUT / f"{name}.html").write_text(page, encoding="utf-8")
    print(name, len(page.encode("utf-8")), "bytes")
