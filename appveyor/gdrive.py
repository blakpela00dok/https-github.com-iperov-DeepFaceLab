#!/usr/bin/env python
import sys
from operator import itemgetter
import requests
import re
import datetime
import os

FOLDER_ID = os.environ['FOLDER_ID']
API_KEY = os.environ['API_KEY']

p = re.compile(r'DeepFaceLab([\w.]+)_build_(\d\d)_(\d\d)_(\d\d\d\d)\.exe')


def get_folder_url(folder_id):
    return "https://www.googleapis.com/drive/v3/files?q='" + folder_id + "'+in+parents&key=" + API_KEY


def get_builds():
    url = get_folder_url(FOLDER_ID)
    r = requests.get(url)
    files = r.json()['files']
    builds = []
    for file in files:
        if file['mimeType'] == 'application/x-msdownload':
            filename = file['name']
            m = p.match(filename)
            if m:
                file['arch'], month, day, year = m.groups()
                file['date'] = datetime.date(int(year), int(month), int(day))
                builds.append(file)
            else:
                # print('Not parsed: ', filename)
                pass
    return builds


def get_latest_build(arch):
    builds = get_builds()
    arch_builds = [build for build in builds if build['arch'] == arch]
    arch_builds.sort(key=itemgetter('date'))
    return arch_builds[-1]


def get_download_url(file_id):
    return "https://www.googleapis.com/drive/v3/files/" + file_id + "?alt=media&key=" + API_KEY


def main():
    arch = sys.argv[1]
    # print(arch)
    latest_build = get_latest_build(arch)
    # print(latest_build)
    download_url = get_download_url(latest_build['id'])
    print(download_url)


if __name__ == "__main__":
    main()
