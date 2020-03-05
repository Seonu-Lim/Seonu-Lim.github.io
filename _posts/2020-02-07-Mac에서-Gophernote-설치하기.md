# 맥에서 gophernotes 설치하기.

1. Terminal 을 엽니다.
2. pkg-config 와 libzmq 를 설치합니다.
  * pkg-config 설치방법 : 터미널에서  `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null` 입력 후 엔터. 중간에 컴퓨터 패스워드 입력해야 함. 모두 실행되면 `brew install pkg-config` 실행.
  * libzmq 설치 방법 : 터미널에서 `brew install zmq` 입력 후 엔터.
3. 다음 코드를 차례로 실행합니다.
```
$env GO111MODULE=on go get github.com/gopherdata/gophernotes
$mkdir -p ~/Library/Jupyter/kernels/gophernotes
$cd ~/Library/Jupyter/kernels/gophernotes
$cp "$(go env GOPATH)"/pkg/mod/github.com/gopherdata/gophernotes@v0.6.1/kernel/*  "."
$sed "s|gophernotes|$(go env GOPATH)/bin/gophernotes|" < kernel.json.in > kernel.json
```
  * 세번째 줄에서 Permission Denied 오류가 난다면, 마지막 줄만 앞에 sudo 를 붙여서 다시 실행시키면 됩니다.
  * 네번째 줄에서 Permission Denied 오류가 난다면, 다음의 코드를 실행한 뒤에 네번째 줄을 재시도해봅시다. 

```
$chmod +w ./kernel.json
```
