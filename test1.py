import http.client, json, time

strs = ['3D']

conn = http.client.HTTPSConnection("api.danmaku.suki.club")
payload = ''
headers = {
    'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
}

with open("wase_liveinfo.json", 'r', encoding='utf-8') as f:
    liveinfos = json.load(f)
    print("Liveinfo loaded.")

n = len(liveinfos['data']['lives'])
outputs = []

for idx, liveinfo in enumerate(liveinfos['data']['lives']):
    print("Finding {}/{}:".format(idx+1, n))
    if liveinfo['startDate'] < 1654347607000:
        break
    else:
        conn.request("GET", "/api/info/live?liveid={}".format(liveinfo['liveId']), payload, headers)
        res = conn.getresponse()
        data_str = res.read().decode('utf-8')
        data_json = json.loads(data_str)
        with open('wase_dm/{}.json'.format(time.strftime('%m-%d', time.localtime(float(liveinfo['startDate'])/1000)))) as f:
            json.dump(data_json, f)
            continue
        for danmaku in data_json['data']['danmakus']:
            if not 'message' in danmaku:
                #print(danmaku)
                #quit()
                continue
            for s in strs:
                if s in danmaku['message']:
                    #print("{}: {}, {}".format(liveinfo['title'], time.strftime('%m-%d %H:%M', time.localtime(float(danmaku['sendDate'])/1000)), danmaku['message']))
                    output = "{}, {}".format(time.strftime('%m-%d %H:%M', time.localtime(float(danmaku['sendDate'])/1000)), danmaku['message'])
                    #print(output)
                    outputs.append(output)
                    #quit()

quit()
with open("wase_dminfo.out", 'w', encoding='utf-8') as f:
    f.writelines(outputs)
    #[print(output) for output in outputs]