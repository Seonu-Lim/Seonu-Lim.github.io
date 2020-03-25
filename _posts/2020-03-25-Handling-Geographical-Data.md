---
title: "우리나라 지도의 위도경도 데이터."
categories:
  - Python
last_modified_at: 2020-03-25T13:00:00+09:00
use_math: false
---

우리나라 지도의 위도 경도 데이터가 필요했던 관계로 구글링을 해봤지만, 내가 찾아낸 shp 파일은 UTM-K 라는 처음 들어보는 좌표계로 작성되어 있었다. ([대한민국 최신 행정구역(SHP) 다운로드 링크](http://www.gisdeveloper.co.kr/?p=2332)) 여기서 시도로 나뉘어진 지도 19년 5월 업데이트 분을 가져왔다.

내 목표는 지도 위 특정 장소들에 점을 찍는 것이었는데, 문제는 점들이 위도, 경도로 되어있다는 점이었다. 물론 점들을 UTM-K 로 변환하면 편하겠지만, 나 포함 다른 사람들이 더 익숙한 위경도 데이터로 바꾸고 싶었다. 그리고 그걸 ~~굳이~~ 파이썬으로 하고 싶었다. 문제는 점 하나 변환하는 건 일도 아니지만 이 데이터는 수많은 Polygon 들로 이루어져 있다는 점이었다. Polygon 한개는 서울 기준으로 8000개가 넘는 점들로 이루어져 있다. 그리고 섬이 많은 전라도 쪽 등은 MultiPolygon 으로 되어 있어서 생각보다 까다로웠다. MultyPolygon 은 Polygon 으로 나눈 뒤 그것을 다시 점으로 나누고 위경도로 변환 후에 다시 Polygon 으로 합치고 그걸 또 MultiPolygon으로 바꾸는 과정을 거쳤다. 먼저 필요한 파이썬 패키지는 아래와 같다.

```Python
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
import shapely.geometry as geom
import shapely.wkt
````


그리고 받아놓은 shp 파일을 데려온다.

```Python
korea = gpd.read_file("CTPRVN_201905")
```

좌표계 두개 (원래 좌표계와 바꿀 좌표계)를 변수에 넣어둔다. 좌표계들의 EPSG 코드를 알아내서 쓰면 된다. 나는 [데이터사이언스 스쿨의 포스트](https://datascienceschool.net/view-notebook/ef921dc25e01437b9b5c532ba3b89b02/) 를 참고하였다.

```Python
inProj = Proj({'init': 'epsg:5179'})
outProj = Proj({'init': 'epsg:4326'})
```

그리고 이제 변환하면 된다.

```Python
for i in range(len(korea.geometry)) :
    region = korea.geometry[i]
    if region.type == "Polygon" :
        coords_obj = list(region.exterior.coords)
        empty_list = []
        for j in range(len(coords_obj)) :
            x,y = transform(inProj,outProj,coords_obj[j][0],coords_obj[j][1])
            empty_list.append(geom.Point(x,y))
        result = geom.Polygon([[p.x,p.y] for p in empty_list])
    elif region.type == "MultiPolygon" :
        pols = list(region)
        for j in range(len(pols)) :
            coords_obj = list(pols[j].exterior.coords)
            empty_list = []            
            for k in range(len(coords_obj)):
                x,y = transform(inProj,outProj,coords_obj[k][0],coords_obj[k][1])
                empty_list.append(geom.Point(x,y))
            empty_list = geom.Polygon([p.x,p.y] for p in empty_list)
            pols[j] = empty_list
        result = geom.MultiPolygon(pols)
    korea.geometry[i] = result
    print(i+1)
print("Done")
```

MultiPolygon 인 경우, polygon 으로 나누는 과정이 추가되어야 한다. 이 코드를 저 파일 넣고 돌리면 24시간은 아니고 한 12시간 걸리는 듯 하다. (자고 일어났는데 다 돌아가 있었다) 소요 시간을 정확히 알고 싶은 사람은 `print(i+1)` 을 지우고 `tqdm`을 사용하도록 하자. 서울은 10분 정도 걸렸는데, 다도해인 전라남도쪽이 미친 MultiPolygon이어서 그랬는지 세상 오래 걸렸다. 그러니까 파이썬으로 변환하는 일은 없도록 하자. 서울처럼 작은 파일이라면 모르겠지만.

아무튼 변환은 잘 되었고, 두고두고 쓰려고 shp파일을 저장해 두었다. 이 블로그를 보는 사람들 중 혹시 필요하다면 굳이 나처럼 고통받지 말고 그냥 변환 되어있는거 쓰라고 변환된 파일의 링크를 올려둔다.

[**korea shp file download**](https://drive.google.com/drive/folders/1zz4vLKTKa280WlqisI2ao2J2O8jt9A3j?usp=sharing)
