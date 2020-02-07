# Mac에서의 Gophernote 설치 (Kernel Error)

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
  * ( *확실하지 않은 해결책* ) 네번째 줄에서 Permission Denied 오류가 난다면, 앞에 sudo를 붙이고 kernel.json.in 양 옆의 뾰족 괄호를 없애면 실행은 되며, 설치 확인을 위해 `$ "$(go env GOPATH)"/bin/gophernotes`를 실행시키면 제대로 설치되었다고 한다. 하지만 현재 블로그 주인은 Kernel Error에 봉착했음. 이전 포스트 참고.
