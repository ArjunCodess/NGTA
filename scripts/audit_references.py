"""Audit publisher-deposited metadata; preserve unresolved entries explicitly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time
import requests


def canonical(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def audit(bibliography, manuscript, output, apply=False):
    content=Path(bibliography).read_text(encoding='utf-8')
    cited=set()
    for value in re.findall(r'\\cite\w*\{([^}]+)\}',Path(manuscript).read_text(encoding='utf-8')):
        cited.update(k.strip() for k in value.split(','))
    entries=[]
    for match in re.finditer(r'@(\w+)\{([^,]+),(.*?)(?=\n@|\Z)',content,re.S):
        fields=dict(re.findall(r'(\w+)\s*=\s*\{(.*?)\}(?=,?\s*\n)',match[3],re.S))
        if match[2] in cited:
            entries.append((match[1],match[2],fields,match[0]))
    def fetch(entry):
        kind,key,fields,block=entry
        result={'key':key,'original':fields,'status':'unresolved','changes':{}}
        if kind not in {'article','inproceedings','incollection','book'}:
            result['reason']='non-publisher source requires repository/author verification'
            return result
        doi=fields.get('doi')
        url='https://api.crossref.org/works/'+doi if doi else 'https://api.crossref.org/works'
        params=None if doi else {'query.title':fields['title'],'rows':5}
        try:
            for attempt in range(3):
                response=requests.get(url,params=params,timeout=30,headers={'User-Agent':'NGTA-bibliography-audit/1.0'})
                if response.status_code not in {429,503}:break
                time.sleep(2+attempt)
            response.raise_for_status()
            message=response.json()['message']
            candidates=[message] if doi else message['items']
            best=max(candidates,key=lambda m:SequenceMatcher(None,canonical(fields['title']),canonical(m.get('title',[''])[0])).ratio())
            score=SequenceMatcher(None,canonical(fields['title']),canonical(best.get('title',[''])[0])).ratio()
            result.update(source='https://api.crossref.org/works/'+best['DOI'],title_match=score,
                          candidate_title=best.get('title',[''])[0])
            if score<.97:
                result['reason']='no sufficiently close publisher title match; no automatic replacement'
                return result
            values={}
            authors=best.get('author',[])
            if authors:
                values['author']=' and '.join(' '.join(filter(None,[a.get('given'),a.get('family')])) if a.get('family') else '{'+a.get('name','')+'}' for a in authors)
            values['doi']=best['DOI']
            for source_field,bib_field in [('volume','volume'),('issue','number'),('page','pages')]:
                if best.get(source_field):values[bib_field]=str(best[source_field]).replace('-', '--') if bib_field=='pages' else str(best[source_field])
            date=best.get('published-print',best.get('published',best.get('published-online')))
            if date:values['year']=str(date['date-parts'][0][0])
            if best.get('container-title') and kind=='article':values['journal']=best['container-title'][0]
            result['verified_metadata']=values
            result['changes']={k:{'before':fields.get(k),'after':v} for k,v in values.items() if fields.get(k)!=v}
            result['status']='publisher_metadata_matched'
        except Exception as error:
            result['reason']=str(error)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(fetch,entries))
    if apply:
        for entry,result in zip(entries,records):
            if result['status']!='publisher_metadata_matched':continue
            kind,key,fields,block=entry
            fields.update(result['verified_metadata'])
            updated='@'+kind+'{'+key+',\n'+',\n'.join('  '+name+' = {'+value+'}' for name,value in fields.items())+'\n}\n'
            content=content.replace(block,updated)
        Path(bibliography).write_text(content,encoding='utf-8',newline='\n')
    report={'scope':'all manuscript-cited entries; metadata audit, not verification of scientific claims',
            'citation_keys':len(cited),'records':records,
            'publisher_matches':sum(r['status']=='publisher_metadata_matched' for r in records),
            'unresolved':[r['key'] for r in records if r['status']!='publisher_metadata_matched']}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'publisher_matches':report['publisher_matches'],'unresolved':report['unresolved']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bibliography',default='paper/references.bib')
    parser.add_argument('--manuscript',default='paper/main.tex')
    parser.add_argument('--output',default='results/research_checks/reference_metadata.json')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    audit(args.bibliography,args.manuscript,args.output,args.apply)
