---
title: "Python에 MariaDB 에 있는 데이터 가져오기"
categories:
  - Python
last_modified_at: 2020-08-12T13:00:00+09:00
use_math: false
---

방학동안 딥러닝 경험을 쌓고자 작은 스타트업에 다니기 시작했다. ~할말은 많지만 하지 않겠다~ 여기서는 데이터베이스를 MariaDB 로 관리하는데, 나는 엄청난 뉴비이기 때문에 MariaDB 에 있는 데이터를 가져오는 것부터 버벅거렸다. 먼저 나는 local은 MacOS, remote는 ubuntu를 사용하고 있음을 먼저 밝힌다. 두 개 다 비슷했는데 둘 다 헤맸다. 큰 틀만 정리하자면

1. MariaDB python Connector 를 설치한다. [여기서 설치](https://mariadb.com/downloads/#connectors)
2. 터미널에 `pip install mariadb`
3. 이제 IDE 를 열고 아래와 같이 코드를 실행한다.

```python
import mariadb
import sys
import pandas as pd

try:
    conn = mariadb.connect(
        user="enter.username",
        password="enter.password",
        host="enter.host.name",
        port=3306,
        database="enter.database.name"
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)
cur = conn.cursor()
cur.execute('select * from TABLE_NAME')
res = cur.fetchall()
data = pd.DataFrame.from_records(res)
```

같은 데이터베이스에서 다른 테이블을 가져오고 싶으면

```python
cur.execute('select * from TABLE_NAME2')
res = cur.fetchall()
data2 = pd.DataFrame.from_records(res)
```
이 부분만 새로 실행하면 된다.
