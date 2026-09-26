import pandas as pd, csv, re, unicodedata
R='/private/tmp/claude-501/-Users-mac/4ff6881f-3025-4335-b379-01a3f99e0553/scratchpad/real/'
e=pd.read_parquet(R+'v4_err_s2.parquet')
need=set(e.s1)
s1={}
with open(R+'train_source1.tsv',newline='') as f:
    r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE); next(r)
    for x in r:
        if x[0] in need: s1[x[0]]=(x[1],x[2])
LEET=str.maketrans({'0':'o','1':'l','3':'e','4':'a','5':'s','7':'t','8':'b','@':'a','$':'s'})
LEGAL={'pvt','private','ltd','limited','llc','l','inc','incorporated','corp','corporation','co','company','sa','sas','sarl','llp','plc','pllc','lp','the','and','of','dba','group','holding','holdings','enterprises','enterprise'}
ABBR={'rd':'road','st':'street','ave':'avenue','av':'avenue','blvd':'boulevard','dr':'drive','ln':'lane','ct':'court','hwy':'highway','nr':'near','opp':'opposite','apt':'apartment','ste':'suite','fl':'floor','pl':'place','sq':'square','mg':'mg','n':'north','s':'south','e':'east','w':'west','r':'rue','bd':'boulevard','null':''}
def base(s):
    s=unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
    return re.findall(r'[a-z0-9@$]+',s)
def cname(s):
    out=[]
    for t in base(s):
        if sum(c.isalpha() for c in t)>=len(t)/2: t=t.translate(LEET)
        t=re.sub(r'[^a-z0-9]','',t)
        if t and t not in LEGAL: out.append(t)
    return tuple(sorted(set(out)))
def caddr(s):
    toks=[]; nums=set()
    for t in base(s):
        t=re.sub(r'[^a-z0-9]','',t)
        if t.isdigit(): nums.add(t.lstrip('0') or '0'); continue
        t=ABBR.get(t,t)
        if t: toks.append(t)
    return frozenset(toks), frozenset(nums)
rows=[]
for x in e.itertuples():
    if x.s1 not in s1: continue
    n1,a1=s1[x.s1]
    ne = cname(n1)==cname(x.rn)
    A1,N1=caddr(a1); A2,N2=caddr(x.ra)
    jac=len(A1&A2)/max(1,len(A1|A2))
    numeq = (N1==N2) and len(N1)>0
    rows.append((x.c,x.y,ne,numeq,jac>=0.8, a1.strip()=='' or x.ra.strip()==''))
d=pd.DataFrame(rows,columns=['c','y','name_eq','num_eq','addr_j80','empty_addr'])
d['canon_exact']=d.name_eq&d.num_eq&d.addr_j80
print('n rows', len(d), '(y=True -> FN rows, y=False -> FP rows)')
print(d.groupby('y')[['name_eq','num_eq','addr_j80','canon_exact','empty_addr']].mean().round(3).to_string())
print(d.groupby(['c','y'])[['canon_exact']].agg(['mean','size']).round(3).to_string())
