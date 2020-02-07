# 1. Intro : 처음 뵙겠습니다.


써본 언어라고는 통계 분석을 위한 Python 과 R 뿐인 무지랭이이므로 천천히 진행을 해본다. Data Input 부분에서 구현할 것들은

* User 가 데이터의 경로를 입력하도록 한다.
* 그러면 그 경로로 가서 데이터를 가져온다.

..뿐인데, 여기서 '데이터를 가져온다' 부분만 해보되, 데이터를 프린트해보도록 하겠다. 대충 검색해서 나온 코드는 아래와 같다. 데이터마이닝 수업에 쓰던 샘플 데이터를 사용했다. 


``` go
package main

import (
	"fmt"
	"io/ioutil"
)

func main() {
	file, err := ioutil.ReadFile("/Users/seonwoolim/Desktop/대학원/2학기/데마/data/harris.dat")
	fmt.Println(string(file))
}
```

구글링으로 찾아와서 그대로 따라한 이 코드의 뜻을 대충 짐작해본다. 
* import야 뭐 패키지 가져온 것이고, io/ioutil 이 편해보여서 가져와봤다.
* main이라는 것이 내가 만들 무언가..인 것 같다. 
* ioutil의 ReadFile 함수로 harris.dat 파일을 가져왔다. 주의할 점은, **ReadFile 함수의 output은 두개라는 점이다**. 두번째 output은 error 이다. 사람에 따라 error 이 없지 않을 경우 (`if err != nil `) 에러 메시지를 출력하거나 프로그램이 멈춰버리도록(`panic`,`log.Fatal` 등) 코드를 추가한 것을 봤지만 나는 귀찮아서 안했다. 
* print를 해보기 위해서 string 으로 변환하였다. 이 부분은 추후에 없어질 예정이다. 

코드를 test.go 라는 이름으로 저장한다. R이나 Python은 한 줄씩 실행이 되지만 go 는 그렇지가 않다. 
심지어 코드 짜다 말고 저장하면 **이 변수는 왜 declare만 하고 쓰진 않은거야!** 하고 나한테 화를 낸다. 
그러니 중간에 코드를 잘 짰는지 보고 싶다면 대충 코드를 마무리한 척 하고 돌려보도록 하자. 

아무튼 저런 코드를 쓰고 저장하면 그냥 test.go 파일이 하나 만들어져 있을 뿐이다.
이 놈을 실행하려면 터미널에서 저 파일이 있는 디렉토리로 간 다음에 go build test.go 하면 된다.
그러면 드디어 실행! 

이 아니고 같은 디렉토리에 exe 파일이 생긴다. ~개빡쳐~ 그걸 실행시켜야 비로소 나의 코드가 실행되는 것이다. 아 참으로 신묘하다. 
결과물은 아래와 같이 예쁘게 프린트가 되더라.


``` 
Last login: Fri Jan 24 12:37:03 on ttys001
(base) seonui-iMac:~ seonwoolim$ /Users/seonwoolim/Desktop/test ; exit;
3900 12 0 1 0
4020 10 44 7 0
4290 12 5 30 0
4380 8 6.2 7 0
4380 8 7.5 6 0
4380 12 0 7 0
#길어서 중간은 생략
6600 15 64 16 1
6600 15 84 33 1
6600 15 215.5 16 1
6840 15 41.5 7 1
6900 12 175 10 1
6900 15 132 24 1
8100 16 54.5 33 1
logout
Saving session...
...copying shared history...
...saving history...truncating history files...
...completed.

[프로세스 완료됨]

```

...Gophernote 를 쓰기로 결정했다. 매번 이 짓을 할 순 없어... 설치는 링크 https://github.com/gopherdata/gophernotes 를 참고했다. ~~그리고 수많은 오류들과 싸우게 되는데...~~

여기서 참고사항 : 나는 mac osx 를 사용하고 있다. 그리고 컴알못이지만 그저 시키는대로 흘러가는대로 해낼 뿐이다.

링크의 첫번째 줄을 실행시켰다.

```
$ env GO111MODULE=on go get github.com/gopherdata/gophernotes
```

그랬더니 pkg-config 가 없어서 안된다고 한다. ~~그게뭔데~~ 그래서 설치를 했다. 설치 방법은 이 링크 참고 : http://macappstore.org/pkg-config/







