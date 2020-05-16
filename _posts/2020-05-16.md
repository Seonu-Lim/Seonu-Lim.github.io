---
title: "알알못 통계학과의 알고리즘 입문: Hash"
categories:
  - Python
last_modified_at: 2020-05-16T13:00:00+09:00
use_math: false
---

## Intro

솔직히 말해서 알고리즘을 공부하려고 마음 먹은 건 꽤 오래 전인데, 전공도 제대로 못하는게 무슨 알고리즘... 이라면서 계속 미루기만 했다.(그렇다고 전공을 열심히 판 것도 아니잖아...) 그런데 석사 졸업도 이젠 정말 코앞이고, 취업 준비를 위해서는 더 이상 미룰 수가 없더라. 그래서 programmers 에서 알고리즘 독학을 시작하였다. 독학이라기엔 좀 주먹구구식인데, 그냥 무작정 문제를 풀어보고 모르면 구글링하면서 통과할 때까지 코드를 수정하고 제출하고를 반복하는 거다.

아무튼 첫 타자는 hash 라는 것이었는데, hash 가 무엇인고 했더니 그냥 python의 dictionary 같은 것이더라. 그런갑다하고 문제를 풀어봤다. 대체로 금방 풀면 1트만에 성공하는데, 오래 걸리는 경우는 몇몇 test case 에서 시간 초과가 떠서 3트 정도까지 해보고 통과할 수 있었다. 각각의 문제 제목을 클릭하면 프로그래머스의 문제 화면으로 넘어간다.

## 1. [완주하지 못한 선수 문제](https://programmers.co.kr/learn/courses/30/lessons/42576)

Hash 문제라고는 하지만 딱히 hash 를 쓴 것 같지 않은데... 레벨 1 문제기도 하고... 이 문제는 고맙게도 input 의 형태가 엄청나게 제한되어 있어서 그걸 이용하면 풀이는 엄청나게 쉽다.

```python
def solution(participant, completion):
    participant.sort()
    completion.sort()
    completion.append(0)
    i = 0
    while participant[i] == completion[i]:
        i += 1
    return participant[i]
```
코드를 말로 풀어쓰면

1. 그냥 participant랑 completion을 알파벳 순서로 정렬하고
2. completion 끝에 0을 붙여서 participant 랑 길이를 같게 만든다.
3. paricipant 랑 completion을 인덱스가 같은 것들끼리 한개씩 비교하다가 다른 것이 나오면 멈추고 달랐던 participant의 이름을 return 한다.

아주 간단한 문제였다.

## 2. [전화번호 목록](https://programmers.co.kr/learn/courses/30/lessons/42577)

```python
def solution(pb) :
    pb.sort(key=len)
    check = [False]
    while not any(check) :
        candidate = pb.pop(0)
        check = [i.startswith(candidate) for i in pb]
        if len(pb) == 0 :
            answer = True
            break
    try : answer
    except NameError : answer = False
    return answer
```

1. input을 길이 순서로 sort 한다. (길이가 상대적으로 짧은 것이 접두사가 될 테니까)
2. check 라는 list를 만들어서, initial value 로 False 를 넣어둔다.
3. check 에 True 가 들어가면 멈추는 while loop 인데,
  * pb에서 가장 짧은 녀석을 pb에서 pop 시켜서
  * 그 녀석으로 시작하는건지 아닌지 여부인 bool 들을 check 에 저장해준다.
  * pb에 있던 element들이 모두 소진되어서 아무것도 없으면, 접두사가 되는 경우가 없다는 의미이므로 answer 에 True를 저장하고 loop를 멈춘다.
4. answer 이 지정되지 않았으면 (즉, pb가 소진되기 전에 while loop 가 멈추었으면) 접두사가 되는 경우가 있다는 의미이므로 answer 에 False 를 저장한다.
5. answer 를 return한다.

## 3. [위장](https://programmers.co.kr/learn/courses/30/lessons/42578)

이 문제는 수식을 세워서 함수로 만들면 되는 문제였다.

수식은 간단한데, 개수에 각각 1을 더하고 전부 곱한 다음, 1을 빼면 된다. 1을 더하는 이유는 '0개 선택' 이라는 옵션이 있기 때문이고, 1을 빼는 이유는 모두 0개를 선택할 수는 없기 때문이다. 이 수식을 알면 세상 간단한 문제인데, 멍하니 풀다가 itertools 쓰고 시간초과 나고 난리도 아니었다. 고등학교 때 배웠던 것도 최대한 써먹어야 한다는 것을 깨달았다.

```python
import numpy as np
from collections import Counter

def solution(clothes) :
    cdict = Counter([j for i,j in clothes])
    items = cdict.values()
    answer = np.prod([i+1 for i in items]) -1
    return answer
```

1. 옷의 종류 별로 몇 개 씩의 옷을 가지고 있는지 센다. Counter 함수는 이를 {element : count} 모양의 dictionary 형태로 return 한다.
2. 개수에 1씩 더하고 모두 곱한 다음 -1 한 값을 return 한다.

## 4. [베스트 앨범](https://programmers.co.kr/learn/courses/30/lessons/42579)

이 문제가 Hash 문제 중 레벨이 제일 높은데, 솔직히 이거보다 위에 있는 문제들이 더 어려웠던 것 같다.

```python
def gettwotops(l) :
    l.sort(key=lambda x:-x[1])
    twol = l[0:2]
    res = [i for i,j in twol]
    return res

def solution(genres, plays):
    gsum = []
    for i in range(len(genres)) :
        gs = sum([plays[j] for j in range(len(genres)) if genres[j] == genres[i]])
        gsum.append(gs)
    uniq = list(set(gsum))
    uniq.sort(reverse=True)
    gendict = {u:[] for u in uniq}
    for i in gendict :
        gendict[i] = gettwotops([ip[j] for j in range(len(gsum)) if gsum[j] == i])
    return sum(gendict.values(),[])
```
왜 gettwotops 함수를 따로 만들었는지 이유는 잘 모르겠다. 뭐 안될 건 없지

1. gettwotops 함수는 list의 두번째 값을 기준으로 descending 하게 sort 한 다음 앞의 두 값을 list 로 return 하는 함수이다.
2. solution 내부의 for loop 는 gsum list 를 생성하는데, 장르별 재생수를 계산해 준다. 단, length 는 genres 와 동일하므로 곡 별로 장르별 재생수가 1:1 대응되는 리스트이다.
3. gsum의 unique value 들을 descending order 로 sort 하여 uniq 로 저장한다.
4. uniq 의 element 를 key로 하는 dictionary 를 만든다. (gendict)
5. gendict 의 key 값 아래에 각각 장르별 top two list를 value 로 저장한다.
6. gendict 의 value 들을 하나의 리스트로 합치고 return 한다.

## Outro : 이 글을 왠지 읽고 있는 분께...

알고리즘을 단 한번도 배워본 적이 없는 알알못의 코드라서, 프로그래머스에서 통과는 했지만 전공자가 보기에는 어디엔가 어설픈 구석이 있을 거예요. 지나가다가 아 이거 이렇게 하는 거 아닌데... 싶은 분은 댓글로 일해라 절해라 감놔라 배놔라 등등의 훈수 너무나도 환영입니다.
