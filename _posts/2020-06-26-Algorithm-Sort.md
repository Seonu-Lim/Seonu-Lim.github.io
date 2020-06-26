---
title: "알알못 통계학과의 알고리즘 입문: Sort"
categories:
  - Python
last_modified_at: 2020-06-26T13:00:00+09:00
use_math: false
---

## Intro

한달 넘게 블로그에 글을 쓰지 못했다. 이제 석사 3학기가 끝나고 방학이 시작되었으니 다시 짬을 내서 블로그를 업데이트 해보려고 한다. Sort 문제는 아주 예전에 풀었었는데, 정리하는 것은 계속 미루고 있었다. 복습 차원에서 정리해보려고 한다. Sorting 은 막상 푸는 것 자체는 어렵지 않았으나 다 풀고 나서 다른사람들의 풀이를 보면 현타가 오는 상황이 계속됐다. 이번에도 각 문제의 제목을 클릭하면 해당 programmers 문제풀이 페이지로 넘어간다.

## 1. [K번째 수](https://programmers.co.kr/learn/courses/30/lessons/42748)

역시나 첫 문제는 가장 쉬운 문제. 그래서 풀이도 엄청나게 짧았다. (그런데 다른 사람 풀이들을 보니 내 풀이가 짧은 편이 아니었다....)

```python
def solution(array, commands):
    answer = []
    for com in commands :
        arr = array[com[0]-1:com[1]]
        arr.sort()
        answer.append(arr[com[2]-1])
    return answer
```
코드를 말로 풀어쓰면

1. commands 안의 원소 com 은 3개의 원소가 있는 list 이다. com 의 내용에 따라 array를 자르고 sort 한 뒤 com[2] 번째 숫자를 answer 에 append 한다.

...이게 다다. 그런데, 다른 분들의 풀이를 열어봤는데 가장 위에 있던 풀이는 단 한줄 짜리여서 조금 마음의 상처를 받았었다.

```python
def solution(array, commands):
    return list(map(lambda x:sorted(array[x[0]-1:x[1]])[x[2]-1], commands))
```
python 의 map 을 활용하면 이렇게나 코드가 짧아질 수 있다는 점... iterative list가 나오면 for 문부터 쓰지 말고 map으로 해결할 수 있는지부터 생각해보자.


## 2. [가장 큰 수](https://programmers.co.kr/learn/courses/30/lessons/42746)

3문제 중 개인적으로 가장 어려웠던 문제였다. 가장 긴 녀석의 길이만큼 다른 놈들을 이어붙일 때, 자기 자신을 이어붙인다는 생각을 못하고 가장 마지막 수만 계속 이어붙였다가 실패가 떠서 대체 왜?!!??!? 실패가 뜨는 건지 괴로워 했다가 나중에 무릎을 탁쳤던 기억이 난다. 그리고 이 문제도 역시 다른 사람의 풀이를 열고 눈물을 흘렸다. 일단 내 풀이는 아래와 같다.

```python
def solution(numbers) :
    if sum(numbers) == 0 :
        return '0'
    else :
        stn = [str(n) for n in numbers]
        getmaxlen = len(max(stn,key=len))
        newstn = []
        index = list(range(len(stn)))
        for n in stn :
            if len(n) < getmaxlen :
                newn = n * 4
                newn = newn[0:getmaxlen]
                newstn.append(newn)
            else : newstn.append(n)
        index.sort(key=lambda k: (newstn[k]),reverse=True)
        sortstn = [stn[i] for i in index]
        res = ''.join(sortstn)
        return res
```

1. 원소의 합이 0인 경우, 즉 모든 원소가 0일 때에는 굳이 function 전체를 실행하지 않아도 된다.(이거때문에 자꾸 효율성 실패 떴었다.) 그래서 맨 처음에 모든 원소가 0일 경우를 집어넣었다.
2. 원소 합이 0이 아닌 경우
  * 원소들을 string으로 변환하여 그 리스트를 stn 이라고 저장해둔다.
  * 이렇게 변환한 원소들 중 가장 길이가 긴 놈의 길이를 getmaxlen이라고 저장해둔다.
  * sorting을 위해 index 를 만든다. 그냥 stn의 길이만큼의 range list 임.
  * stn의 원소(n)에 대해 for loop를 돌린다.
    - n의 길이가 getmaxlen보다 짧을 경우, n을 4번 반복해서 길이를 늘려버린 다음에 getmaxlen 만큼으로 자르고 newn이라고 저장하고 newstn 에 append 한다.
    - n의 길이가 이미 getmaxlen 일 경우, 그냥 바로 newstn에 append 한다.
  * newstn을 기준으로 index 를 decreasing order 로 sort 한다.
  * index 순서로 stn을 불러오면 sorting 된 stn이 된다.
  * 이걸 하나의 string 으로 붙여서 return 한다.

이제 다른 사람들의 풀이를 보자.

```python
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x: x*3, reverse=True)
    return str(int(''.join(numbers)))
```
ㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 이게 된다는게 진짜 어이가 없었다. 나만 머리 안 돌아가는 더러운 세상... 일단 getmaxlen 길이로 자를 필요도 없었던 거다. 그리고 몇번 반복할지는 그냥 엿장수 마음대로인건가 나도 그냥 4번으로 놓고 풀었지만 약간 의문... 이게 분명 어떤 경우에서는 저 반복횟수가 return값에 영향을 줄 것 같아서 신경이 쓰인다.

## 3. [H-Index](https://programmers.co.kr/learn/courses/30/lessons/42747)



```python
def solution(citations):
    citations.sort()
    answer = []
    for n in range(len(citations)+1) :
        if sum([i>=n for i in citations]) >= n :
            answer.append(n)
    return max(answer)
```

1. citation을 sort 한다.
2. 논문 수에 대해 for loop를 돌린다.
  * 인용 수가 논문 수보다 많거나 같은 논문의 인용수들을 합친 것이 논문 수보다 많거나 같다면, 논문 수를 answer에 append 한다.
3. 이렇게 answer 에 저장 된 논문 수 중 가장 큰 것을 return 한다.


## Outro.

Sort 문제들 푸는 내내 sort 함수를 아주 잘 썼는데, 다른 문제들에서는 sort 를 너무 남발하면 효율성 실패가 뜨는 경우가 꽤 있었다. sort 의 시간복잡도가 O(nlogn) 이라고 하니, 만약 다른 알고리즘 문제인데 sorting 을 해야 한다면 sort 함수에 너무 의존하지 말고 다른 방법은 없을지 고민해보자. Python의 주요 함수의 시간 복잡도를 정리해 둔 링크를 찾았다. 효율성 문제가 있을 때 아래 링크를 확인해보는 것도 좋을 것 같다.

#[Complexity of Python Operations](https://www.ics.uci.edu/~pattis/ICS-33/lectures/complexitypython.txt)
